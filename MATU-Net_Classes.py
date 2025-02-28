import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import SWALR
# import wandb  # for experiment tracking
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path
import argparse
import gc
from sklearn.metrics import r2_score, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import WeightedRandomSampler
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Constants
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = 1
BASE_CHANNELS = 64
# WHAT = "Loss_N_L1"
WHAT_OPS = ["", "focal", "class_balanced"]

def setup_logging(experiment_dir):
    Path(experiment_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{experiment_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Model Components
class SidewalkTransformer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels+1, 32, 7, padding=3),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        # Initialize to identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, x, mask):
        combined = torch.cat([x, mask], dim=1)
        xs = self.localization(combined)
        xs = xs.view(-1, 64 * 16 * 16)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # Apply zoom to focus on sidewalk region
        grid = F.affine_grid(theta, x.size())
        x_transformed = F.grid_sample(x, grid)
        mask_transformed = F.grid_sample(mask, grid)
        
        return x_transformed, mask_transformed

class ShadowFeatureEnhancer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Shadow-specific filters (edges, gradients, etc.)
        self.edge_filters = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.gradient_x = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        self.gradient_y = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels)
        
        # Initialize with specific edge detection kernels
        with torch.no_grad():
            for i in range(channels):
                # Sobel-like filters
                self.gradient_x.weight[i,0,0,:] = torch.tensor([-1.0, 0.0, 1.0])
                self.gradient_y.weight[i,0,:,0] = torch.tensor([-1.0, 0.0, 1.0])
        
        self.combine = nn.Conv2d(channels*3, channels, 1)
        
    def forward(self, x):
        edges = self.edge_filters(x)
        grad_x = self.gradient_x(x)
        grad_y = self.gradient_y(x)
        
        enhanced = torch.cat([edges, grad_x, grad_y], dim=1)
        return self.combine(enhanced)
    
class ContextShadowReasoning(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.context_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, dilation=1)
        self.wide_context = nn.Conv2d(channels, channels, kernel_size=5, padding=6, dilation=3)
        self.combine = nn.Conv2d(channels*2, channels, 1)
        
        # Reasoning components
        self.query_conv = nn.Conv2d(channels, channels//8, 1)
        self.key_conv = nn.Conv2d(channels, channels//8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x, mask):
        # Extract multi-scale context
        local_context = self.context_conv(x)
        wide_context = self.wide_context(x)
        multi_context = self.combine(torch.cat([local_context, wide_context], dim=1))
        
        # Non-local reasoning about shadow patterns
        b, c, h, w = multi_context.shape
        query = self.query_conv(multi_context).view(b, -1, h*w).permute(0, 2, 1)
        key = self.key_conv(multi_context).view(b, -1, h*w)
        value = self.value_conv(multi_context).view(b, -1, h*w)
        
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)
        
        # Apply mask-based weighting
        mask_resized = F.interpolate(mask, size=(h, w), mode='nearest')
        return out * (1 + mask_resized)  # Emphasize sidewalk region

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class MaskedAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_attention = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')
        combined = torch.cat([x, mask], dim=1)
        # print("MaskedAttention shapes:")
        # print(f"x: {x.shape}")
        # print(f"mask: {mask.shape}")
        # print(f"combined: {combined.shape}")
        attention_weights = self.conv_attention(combined)
        return x * attention_weights

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        return x

class OutputHead(nn.Module):
    def __init__(self, in_channels, num_bins=10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1, num_bins)
        
    def forward(self, x, mask):
        features = self.conv(x)
        pooled = self.global_pool(features * mask).squeeze(-1).squeeze(-1)
        logits = self.classifier(pooled)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        # Apply log_softmax for numerical stability
        log_prob = F.log_softmax(inputs, dim=1)
        prob = torch.exp(log_prob)
        # Gather the probabilities of the target classes
        targets = targets.view(-1, 1)  # Add dimension to match with gather operation
        prob_target = prob.gather(1, targets).view(-1)
        log_prob_target = log_prob.gather(1, targets).view(-1)
        # Calculate focal weight
        focal_weight = (1 - prob_target) ** self.gamma
        # Apply alpha if specified
        if self.alpha is not None:
            alpha_target = self.alpha.gather(0, targets.view(-1))
            focal_weight = focal_weight * alpha_target
        loss = -focal_weight * log_prob_target
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, gamma=0.5):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        # Calculate weights
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.weights = torch.FloatTensor(weights)
        
    def forward(self, inputs, targets):
        # Use our fixed FocalLoss implementation
        device = inputs.device
        self.weights = self.weights.to(device)
        # Apply log_softmax for numerical stability
        log_prob = F.log_softmax(inputs, dim=1)
        prob = torch.exp(log_prob)
        # Gather the probabilities of the target classes
        targets = targets.view(-1, 1)  # Add dimension to match with gather operation
        prob_target = prob.gather(1, targets).view(-1)
        log_prob_target = log_prob.gather(1, targets).view(-1)
        # Apply class weights
        weights_target = self.weights.gather(0, targets.view(-1))
        # Calculate focal weight
        focal_weight = (1 - prob_target) ** self.gamma
        # Combine weights
        combined_weight = weights_target * focal_weight
        loss = -combined_weight * log_prob_target
        return loss.mean()
    

# Main Model
class MATU_Net(nn.Module):
    def __init__(self, input_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(input_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels*2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4)
        self.enc4 = ConvBlock(base_channels*4, base_channels*8)
        
        # Attention Modules
        self.attention1 = MaskedAttentionModule(base_channels)
        self.attention2 = MaskedAttentionModule(base_channels*2)
        self.attention3 = MaskedAttentionModule(base_channels*4)
        self.attention4 = MaskedAttentionModule(base_channels*8)
        
        # Transformers
        self.transformer1 = TransformerEncoder(base_channels)
        self.transformer2 = TransformerEncoder(base_channels*2)
        self.transformer3 = TransformerEncoder(base_channels*4)
        self.transformer4 = TransformerEncoder(base_channels*8)
        
        # Decoder
        self.dec4 = ConvBlock(base_channels*12, base_channels*6)
        self.dec3 = ConvBlock(base_channels*8, base_channels*3)
        self.dec2 = ConvBlock(base_channels*4, base_channels)
        self.dec1 = ConvBlock(base_channels*2, base_channels)
        
        # Output
        self.output_head = OutputHead(base_channels)
        self.pool = nn.MaxPool2d(2)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        # Encoder path
        # print("Encoder path")
        # print(f"Input shape: {x.shape}")
        e1 = self.enc1(x)
        # print(f"e1 shape: {e1.shape} From Input Channels: 3, to Base Channels: {BASE_CHANNELS}")
        e1_att = self.attention1(e1, mask)
        # print(f"e1_att shape: {e1_att.shape} From Base Channels: {BASE_CHANNELS} to Base Channels: {BASE_CHANNELS}")
        e1_trans = self.transformer1(e1_att)
        # print(f"e1_trans shape: {e1_trans.shape} From Base Channels: {BASE_CHANNELS} to Base Channels: {BASE_CHANNELS}")
        p1 = self.pool(e1_trans)
        
        e2 = self.enc2(p1)
        e2_att = self.attention2(e2, mask)
        e2_trans = self.transformer2(e2_att)
        p2 = self.pool(e2_trans)
        
        e3 = self.enc3(p2)
        e3_att = self.attention3(e3, mask)
        e3_trans = self.transformer3(e3_att)
        p3 = self.pool(e3_trans)
        
        e4 = self.enc4(p3)
        e4_att = self.attention4(e4, mask)
        e4_trans = self.transformer4(e4_att)
        
        # Decoder path
        # print("\nDecoder shapes:")
        # print(f"d4 - e4_trans: {e4_trans.shape}")
        # print(f"d4 - e3_trans: {e3_trans.shape}")
        d4 = self.dec4(torch.cat([
            F.interpolate(e4_trans, size=e3_trans.shape[2:], mode='bilinear', align_corners=True),
            e3_trans
        ], dim=1))
        # print(f"d4 concat result: {d4.shape}")
        
        # print(f"d3 - d4: {d4.shape}")
        # print(f"d3 - e2_trans: {e2_trans.shape}")
        d3 = self.dec3(torch.cat([
            F.interpolate(d4, size=e2_trans.shape[2:], mode='bilinear', align_corners=True),
            e2_trans
        ], dim=1))
        # print(f"d3 concat result: {d3.shape}")
        
        # print(f"d2 - d3: {d3.shape}")
        # print(f"d2 - e1_trans: {e1_trans.shape}")
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e1_trans.shape[2:], mode='bilinear', align_corners=True),
            e1_trans
        ], dim=1))
        # print(f"d2 concat result: {d2.shape}")
        
        # print(f"d1 - d2: {d2.shape}")
        # print(f"d1 - e1_trans: {e1_trans.shape}")
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=True),
            e1_trans
        ], dim=1))
        # print(f"d1 concat result: {d1.shape}")
        
        return self.output_head(d1, mask)

# class Enhanced_MATU_Net(nn.Module):
#     def __init__(self, input_channels=3, base_channels=64):
#         super().__init__()
        
#         # Sidewalk-centric spatial transformer
#         self.sidewalk_transformer = SidewalkTransformer(input_channels)
        
#         # Shadow-specific feature enhancers for each encoder level
#         self.shadow_enhance1 = ShadowFeatureEnhancer(base_channels)
#         self.shadow_enhance2 = ShadowFeatureEnhancer(base_channels*2)
#         self.shadow_enhance3 = ShadowFeatureEnhancer(base_channels*4)
#         self.shadow_enhance4 = ShadowFeatureEnhancer(base_channels*8)
        
#         # Context reasoning modules
#         self.context_reason1 = ContextShadowReasoning(base_channels)
#         self.context_reason2 = ContextShadowReasoning(base_channels*2)
        
#         # Rest of your MATU-Net architecture
#         # ...

#     def forward(self, x, mask):
#         # Apply sidewalk-centric transformation
#         x_focused, mask_focused = self.sidewalk_transformer(x, mask)
        
#         # Encoder path with shadow enhancement
#         e1 = self.enc1(x_focused)
#         e1 = self.shadow_enhance1(e1)
#         e1_att = self.attention1(e1, mask_focused)
#         e1_trans = self.transformer1(e1_att)
#         e1_context = self.context_reason1(e1_trans, mask_focused)
#         p1 = self.pool(e1_context)
        
#         # Continue with enhanced features
#         # ...

# Custom Dataset
class ShadowDataset(Dataset):
    def __init__(self, images, masks, targets, num_bins=10):
        """
        Args:
            images: Sentinel images (N, 3, 32, 32)
            masks: Sidewalk masks (N, 1, 32, 32)
            targets: Shadow percentages (N, 1)
        """
        self.images = images  
        self.masks = masks    
        print(f"Dataset init targets: min={targets.min()}, max={targets.max()}, dtype={targets.dtype}")
        bins = torch.linspace(0, 100, num_bins+1)
        self.targets = torch.bucketize(targets.squeeze(), bins, right=False).clamp(max=num_bins-1).long()
        print(f"After bucketize: min={self.targets.min()}, max={self.targets.max()}, dtype={self.targets.dtype}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        target = self.targets[idx]
        return image, mask, target
    
def load_data(WHAT, images_path, masks_path, targets_file):
    print("Loading data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    images = []
    masks = []
    valid_indices = []
    targets = np.load(targets_file)
    # targets = targets/100
    targets = np.clip(targets, 0, 100)
    print(f"After normalization: min={targets.min()}, max={targets.max()}, dtype={targets.dtype}")
    if WHAT == "Log-transform":
        targets = np.log1p(targets)
    for i in range(len(targets)):
        # Load image
        img_path = os.path.join(images_path, f'patch_{i}.png')
        mask_path = os.path.join(masks_path, f'mask_{i}.npy')
        if not os.path.exists(img_path):
            print(f"Skipping index {i}: files not found")
            continue
        img = Image.open(img_path)
        img = transform(img)
        mask = np.load(mask_path)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        images.append(img)
        masks.append(mask)
        valid_indices.append(i)
    images = torch.stack(images)
    masks = torch.stack(masks)
    targets = torch.from_numpy(targets[valid_indices]).float().unsqueeze(1)
    print(f"Final targets: min={targets.min()}, max={targets.max()}, dtype={targets.dtype}")
    print(f"Final targets with Long(): min={targets.long().min()}, max={targets.long().max()}, dtype={targets.long().dtype}")
    return images, masks, targets.long()

# Loss Function
class MaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        loss = self.loss_fn(pred, target)
        return loss.mean()

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        weights = torch.exp(target * 5) # Higher weights for larger values
        loss = (weights * (pred - target) ** 2).mean()
        return loss
    
# Loss Function
class Loss_N_L1(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        loss = self.loss_fn(pred, target)
        weighted_mse = WeightedMSELoss()
        loss = 0.8 * weighted_mse(pred, target) + 0.2 * F.l1_loss(pred, target)
        return loss.mean()

def plot_learning_curves(train_losses, val_losses, experiment_dir):
   plt.figure(figsize=(10, 6))
   plt.plot(train_losses, label='Training Loss')
   plt.plot(val_losses, label='Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Learning Curves')
   plt.legend()
   plt.grid(True)
   plt.savefig(f'{experiment_dir}/learning_curves.png')
   plt.close()

def lr_range_test(model, train_loader, criterion, device, min_lr=1e-5, max_lr=1e-1, beta=0.97):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=min_lr)
    num_batches = len(train_loader)
    mult = (max_lr/min_lr)**(1/num_batches)
   
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    lrs = []
   
    model.train()
    for images, masks, targets in train_loader:
        batch_num += 1
        current_lr = min_lr * (mult ** batch_num)
        for g in optimizer.param_groups:
            g['lr'] = current_lr
           
        images = images.to(device)
        masks = masks.to(device) 
        targets = targets.to(device)
       
        optimizer.zero_grad()
        logits = model(images, masks)
        loss = criterion(logits, targets)
       
        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
       
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
            
        if batch_num == 1 or smoothed_loss < best_loss:
            best_loss = smoothed_loss
           
        losses.append(smoothed_loss)
        lrs.append(current_lr)
       
        loss.backward()
        optimizer.step()
   
    return lrs, losses

def predict_and_plot_checkpoint(model_path, train_dataset, device, WHAT=""):
    # Load model
    model = MATU_Net(input_channels=3, base_channels=BASE_CHANNELS)
    model.load_state_dict(torch.load(f"{model_path}/best_model.pth", weights_only=True))
    model.to(device)
    model.eval()
    
    # Create dataloader
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, masks, target in loader:
            images = images.to(device)
            masks = masks.to(device)
            _, shadow_percentage = model(images, masks)
            predictions.extend(shadow_percentage.cpu().numpy())
            targets.extend(target.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Transform back if using log transform
    if WHAT == "Log-transform":
        predictions = np.expm1(predictions)
        targets = np.expm1(targets)
    
    # Calculate R2 score
    r2 = r2_score(targets, predictions)
    
    # Create scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.xlabel('True Shadow Percentage')
    plt.ylabel('Predicted Shadow Percentage')
    plt.title(f'Predictions vs Targets (R² = {r2:.3f})')
    plt.tight_layout()
    plt.savefig(f'{model_path}/prediction_scatter.png')
    plt.close()
    
    return predictions, targets, r2

def calculate_class_weights(loader):
    all_targets = []
    for _, _, targets in loader:
        all_targets.extend(targets.numpy())
    counts = np.bincount(all_targets)
    weights = 1. / counts
    return torch.FloatTensor(weights)

def create_balanced_sampler(targets, num_bins=10):
    bins = torch.linspace(0, 100, num_bins+1)
    targets = torch.bucketize(targets.squeeze(), bins, right=False).clamp(max=num_bins-1).long()
    class_counts = torch.bincount(targets)
    print(f"Bins: {bins}")
    print(f"Class counts: {class_counts}")
    weights = 1 / class_counts[targets]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

# Training Function
def train_model(WHAT, shp_path, task_id, model, optimal_lr, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    experiment_dir = f'{shp_path}/model_results/{WHAT}_Exp_Class_{task_id}_{int(time.time())}'
    logger = setup_logging(experiment_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimal_lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )

    if WHAT == 'focal':
        criterion = FocalLoss(alpha=calculate_class_weights(train_loader).to(device), gamma=2.0)
    elif WHAT == 'class_balanced':
        # Get counts per class
        class_counts = torch.bincount(torch.cat([targets for _, _, targets in train_loader]))
        criterion = ClassBalancedLoss(samples_per_class=class_counts.tolist(), beta=0.9999, gamma=0.5)
    else:  # default to cross entropy
        criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(train_loader).to(device))
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        try:
            # Training
            model.train()
            train_loss = 0
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for batch_idx, (images, masks, targets) in enumerate(pbar):
                    images, masks, targets = images.to(device), masks.to(device), targets.to(device)
                    optimizer.zero_grad()
                    logits = model(images, masks)
                    loss = criterion(logits, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    train_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
            
            # Validation
            model.eval()
            val_loss = 0
            train_predictions, train_targets_list = [], []
            val_predictions, val_targets_list = [], []

            with torch.no_grad():
                for images, masks, targets in train_loader:
                    images, masks, targets = images.to(device), masks.to(device), targets.to(device)
                    logits = model(images, masks)
                    pred_bins = torch.argmax(logits, dim=1)
                    train_predictions.extend(pred_bins.cpu().numpy())
                    train_targets_list.extend(targets.cpu().numpy())

                for images, masks, targets in val_loader:
                    images, masks, targets = images.to(device), masks.to(device), targets.to(device)
                    logits = model(images, masks)
                    pred_bins = torch.argmax(logits, dim=1)
                    loss = criterion(logits, targets)
                    val_loss += loss.item()
                    val_predictions.extend(pred_bins.cpu().numpy())
                    val_targets_list.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            logger.info(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

            if WHAT == "Log-transform":
                # In your training loop where you calculate R2:
                train_predictions = np.array(train_predictions)
                val_predictions = np.array(val_predictions)
                train_targets_list = np.array(train_targets_list)
                val_targets_list = np.array(val_targets_list)

                train_predictions = np.expm1(train_predictions)
                val_predictions = np.expm1(val_predictions)
                train_targets_list = np.expm1(train_targets_list) 
                val_targets_list = np.expm1(val_targets_list)

            train_r2 = r2_score(train_targets_list, train_predictions)
            val_r2 = r2_score(val_targets_list, val_predictions)
            logger.info(f'Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}')
            train_accuracy = accuracy_score(train_targets_list, train_predictions)
            val_accuracy = accuracy_score(val_targets_list, val_predictions)
            logger.info(f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
            train_f1_macro = f1_score(train_targets_list, train_predictions, average='macro')
            val_f1_macro = f1_score(val_targets_list, val_predictions, average='macro')
            logger.info(f'Train F1 macro: {train_f1_macro:.4f}, Val F1 macro: {val_f1_macro:.4f}')
            train_f1_weighted = f1_score(train_targets_list, train_predictions, average='weighted')
            val_f1_weighted = f1_score(val_targets_list, val_predictions, average='weighted')
            logger.info(f'Train F1 Weighted: {train_f1_weighted:.4f}, Val F1 Weighted: {val_f1_weighted:.4f}')
            all_targets, all_preds = list(train_targets_list) + list(val_targets_list), list(train_predictions) + list(val_predictions)
            r2 = r2_score(all_targets, all_preds)
            accuracy = accuracy_score(all_targets, all_preds)
            f1_macro = f1_score(all_targets, all_preds, average='macro')
            f1_weighted = f1_score(all_targets, all_preds, average='weighted')
            logger.info(f'Overall R2: {r2:.4f}, Overall Accuracy: {accuracy:.4f}, Overall F1 macro: {f1_macro:.4f}, Overall F1 weighted: {f1_weighted:.4f}\n\n')
            # Plot learning curves
            plot_learning_curves(train_losses, val_losses, experiment_dir)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Create scatter plot
                plt.figure(figsize=(10, 10))
                plt.scatter(all_targets, all_preds, alpha=0.5)
                plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
                plt.xlabel('True Shadow Percentage', fontsize=16)
                plt.ylabel('Predicted Shadow Percentage', fontsize=16)
                plt.title(f'Predictions vs Targets (R² = {r2:.3f}, Accuracy = {accuracy:.4f},\nF1 macro = {f1_macro:.4f}, F1 weighted = {f1_weighted:.4f})', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{experiment_dir}/prediction_scatter.png')
                plt.close()


                torch.save(model.state_dict(), f'{experiment_dir}/best_model.pth')
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, f'{experiment_dir}/checkpoint_epoch_{epoch+1}.pth')
        except Exception as e:
            logger.error(f'Error in epoch {epoch}: {str(e)}')
            continue

def main():
    print(f"Job {os.getenv('SLURM_JOB_ID')} GPU Memory Usage:")
    print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    task_id = args.seed
    WHAT = WHAT_OPS[task_id-1]
    # WHAT = "Classes"

    shp_path = 'C:/Users/mittal53/Box/Shade Maps/data/'
    shp_path = '/scratch/gilbreth/mittal53/ShadeMaps/data/'
    city = 'LA_test_20K'
    images_path = f'{shp_path}/Training_Datasets/{city}/sidewalk_images/'
    masks_path = f'{shp_path}/Training_Datasets/{city}/sidewalk_masks/'
    targets_path = f"{shp_path}/Training_Datasets/{city}/{city}_percent_shade.npy"

    images, masks, targets = load_data(WHAT, images_path, masks_path, targets_path)
    # split_idx = int(len(images))-1
    # Split data (80-20 split)
    split_idx = int(0.8 * len(images))
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_masks, val_masks = masks[:split_idx], masks[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]

    train_dataset = ShadowDataset(train_images, train_masks, train_targets)
    print(f"Train dataset size: {len(train_targets)}, {train_targets.shape}, {train_targets.dtype}, {train_targets.min()}, {train_targets.max()}")
    train_sampler = create_balanced_sampler(train_targets)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_dataset = ShadowDataset(val_images, val_masks, val_targets)
    val_sampler = create_balanced_sampler(val_targets)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = MATU_Net(input_channels=3, base_channels=BASE_CHANNELS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Task {task_id} using device: {device}")

    # ############## Added things
    # model = model.to(device)
    # criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(train_loader).to(device))
    # lrs, losses = lr_range_test(model, train_loader, criterion, device)
    
    # loss_gradients = np.gradient(losses)
    # optimal_lr = lrs[np.argmin(loss_gradients)]  # Find steepest descent
    # print(f"Optimal learning rate: {optimal_lr}")

    # plt.figure(figsize=(10, 6))
    # plt.plot(lrs, losses)
    # plt.xscale('log')
    # plt.xlabel('Learning Rate')
    # plt.ylabel('Loss')
    # plt.title('Learning Rate Range Test')
    # plt.axvline(x=optimal_lr, color='r', linestyle='--', label=f'Optimal LR: {optimal_lr:.4f}')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'lr_range_test_{WHAT}_{task_id}.png')
    # plt.close()

    # ############## End of added things
    ## Values come from experiment_1 - lr_range_test_1.png
    # optimal_lr = 4e-3

    # WHAT_OPS = ["", "WeightedMSELoss", "Loss_N_L1"]
    optimal_lrs = [3e-2, 1e-2, 2e-2]
    optimal_lr = optimal_lrs[task_id-1]
    train_model(WHAT, shp_path, task_id, model, optimal_lr, train_loader, val_loader)

    # Usage
    # predictions, targets, r2 = predict_and_plot_checkpoint(
    #     model_path='/scratch/gilbreth/mittal53/ShadeMaps/data/model_results/_experiment_1_1740151305/',
    #     train_dataset=train_dataset,
    #     device=device
    # )

if __name__ == "__main__":
    main()