import os
import time
import copy
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as TF

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Define the classes
CLASSES = [
    'aphid', 'fungus', 'leaf_blight1', 'leaf_blight2', 
    'leaf_blight_bacterial1', 'leaf_blight_bacterial2',
    'leaf_spot1', 'leaf_spot2', 'leaf_spot3', 'leaf_yellow', 
    'mosaic', 'ragged_stunt', 'virus', 'worm'
]

# Configuration parameters
class Config:
    data_dir = 'B:\DeepLearning\dataset'  # Change this to your dataset path
    img_size = 180
    batch_size = 16
    num_workers = 6
    num_epochs = 15
    learning_rate = 6e-4
    weight_decay = 5e-4
    device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'models'
    log_dir = 'logs'
    early_stopping_patience = 10
    class_weights = False  # Set to True to use class weights
    checkpoint_freq = 5    # Save checkpoint every N epochs
    resume_training = True # Set to False to start fresh training
    
    # Added vegetation filtering parameters
    max_filtering_epochs = 10  # Reduce from 30 to 10 to fit within 15 total epochs 
    masking_output_dir = "masked_images"  # Directory to save masked images

# Make sure save directories exist
for directory in [Config.save_dir, Config.log_dir]:
    os.makedirs(directory, exist_ok=True)

class CustomAugmentation:
    """Advanced data augmentation with vegetation masking that evolves over epochs"""
    def __init__(self, img_size=640, prob=0.5, output_dir="masked_images",
                 current_epoch=0, max_filtering_epochs=30, total_epochs=100,
                 # Thresholds for green vegetation
                 green_g_min=100, green_br_max_ratio=1.5,
                 # Thresholds for brown soil/wood
                 brown_r_min=120, brown_g_min=60, brown_b_max=80,
                 # Thresholds for grey-brown vegetation
                 grey_brown_r_min=100, grey_brown_g_min=80, grey_brown_b_min=60, grey_brown_max_diff=40,
                 # Thresholds for olive/yellow-green vegetation
                 olive_r_min=80, olive_g_min=80, olive_r_g_diff_max=40, olive_b_max=60,
                 # Thresholds for human skin (to mask out)
                 skin_r_min=140, skin_r_max=220, skin_g_min=100, skin_g_max=180, skin_b_min=80, skin_b_max=160,
                 # Thresholds for blue (sky, water, objects - to mask out)
                 blue_b_min=120, blue_bg_ratio_min=1.5, blue_br_ratio_min=1.4,
                 # Black area threshold for crop avoidance
                 black_threshold=50, max_black_percentage=0.5):
        
        self.img_size = img_size
        self.base_prob = prob  # Store base probability for scaling
        self.output_dir = output_dir
        
        # Epochs configuration
        self.current_epoch = current_epoch
        self.max_filtering_epochs = max_filtering_epochs
        self.total_epochs = total_epochs
        
        # Store initial (strictest) threshold values
        # Plant color thresholds
        # Green vegetation
        self.initial_green_g_min = green_g_min
        self.initial_green_br_max_ratio = green_br_max_ratio
        
        # Brown soil/wood
        self.initial_brown_r_min = brown_r_min
        self.initial_brown_g_min = brown_g_min
        self.initial_brown_b_max = brown_b_max
        
        # Grey-brown vegetation
        self.initial_grey_brown_r_min = grey_brown_r_min
        self.initial_grey_brown_g_min = grey_brown_g_min
        self.initial_grey_brown_b_min = grey_brown_b_min
        self.initial_grey_brown_max_diff = grey_brown_max_diff
        
        # Olive/yellow-green vegetation
        self.initial_olive_r_min = olive_r_min
        self.initial_olive_g_min = olive_g_min
        self.initial_olive_r_g_diff_max = olive_r_g_diff_max
        self.initial_olive_b_max = olive_b_max
        
        # Colors to mask out
        # Skin tones
        self.initial_skin_r_min = skin_r_min
        self.initial_skin_r_max = skin_r_max
        self.initial_skin_g_min = skin_g_min
        self.initial_skin_g_max = skin_g_max
        self.initial_skin_b_min = skin_b_min
        self.initial_skin_b_max = skin_b_max
        
        # Blue colors (sky, water, etc.)
        self.initial_blue_b_min = blue_b_min
        self.initial_blue_bg_ratio_min = blue_bg_ratio_min
        self.initial_blue_br_ratio_min = blue_br_ratio_min
        
        # Black region avoidance for cropping
        self.black_threshold = black_threshold
        self.max_black_percentage = max_black_percentage
        
        # Set current thresholds based on epoch
        self.update_thresholds()
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the random erasing transform separately
        self.random_erasing = transforms.RandomErasing(
            p=self.get_augmentation_probability(),
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value='random'
        )
    
    def update_epoch(self, epoch):
        """Update the current epoch and recalculate thresholds"""
        self.current_epoch = epoch
        self.update_thresholds()
        # Update random erasing probability
        self.random_erasing.p = self.get_augmentation_probability()
    
    def get_filtering_factor(self):
        """
        Calculate a factor that decreases from 1.0 to 0.0 over the course of max_filtering_epochs
        1.0 = strongest filtering (strictest thresholds)
        0.0 = no filtering (most permissive thresholds)
        """
        if self.current_epoch >= self.max_filtering_epochs:
            return 0.0  # No filtering after max_filtering_epochs
        
        return 1.0 - (self.current_epoch / self.max_filtering_epochs)
    
    def get_augmentation_probability(self):
        """
        Calculate augmentation probability that increases with epochs
        Starts with self.prob/2 and increases to self.prob
        """
        if self.current_epoch >= self.max_filtering_epochs:
            return self.base_prob  # Full probability after max_filtering_epochs
        
        # Linear increase from prob/2 to prob
        return (self.base_prob / 2) + (self.base_prob / 2) * (self.current_epoch / self.max_filtering_epochs)
    
    def update_thresholds(self):
        """Update all thresholds based on current epoch"""
        factor = self.get_filtering_factor()
        
        # The closer factor is to 1, the stricter we make the thresholds
        # The closer factor is to 0, the more lenient we make the thresholds
        
        # Update plant color thresholds
        # Green vegetation (higher g_min is stricter, lower br_max_ratio is stricter)
        self.green_g_min = self.initial_green_g_min * factor + 50 * (1 - factor)  # Floor at 50
        self.green_br_max_ratio = self.initial_green_br_max_ratio * factor + 3.0 * (1 - factor)  # Ceiling at 3.0
        
        # Brown soil/wood (higher mins are stricter, lower max is stricter)
        self.brown_r_min = self.initial_brown_r_min * factor + 70 * (1 - factor)
        self.brown_g_min = self.initial_brown_g_min * factor + 30 * (1 - factor)
        self.brown_b_max = self.initial_brown_b_max * factor + 120 * (1 - factor)
        
        # Grey-brown vegetation
        self.grey_brown_r_min = self.initial_grey_brown_r_min * factor + 70 * (1 - factor)
        self.grey_brown_g_min = self.initial_grey_brown_g_min * factor + 50 * (1 - factor)
        self.grey_brown_b_min = self.initial_grey_brown_b_min * factor + 30 * (1 - factor)
        self.grey_brown_max_diff = self.initial_grey_brown_max_diff * factor + 80 * (1 - factor)
        
        # Olive/yellow-green vegetation
        self.olive_r_min = self.initial_olive_r_min * factor + 50 * (1 - factor)
        self.olive_g_min = self.initial_olive_g_min * factor + 50 * (1 - factor)
        self.olive_r_g_diff_max = self.initial_olive_r_g_diff_max * factor + 80 * (1 - factor)
        self.olive_b_max = self.initial_olive_b_max * factor + 120 * (1 - factor)
        
        # Colors to mask out - gradually stop masking these out
        # Skin tones
        self.skin_r_min = self.initial_skin_r_min * factor
        self.skin_r_max = self.initial_skin_r_max * factor + 255 * (1 - factor)
        self.skin_g_min = self.initial_skin_g_min * factor
        self.skin_g_max = self.initial_skin_g_max * factor + 255 * (1 - factor)
        self.skin_b_min = self.initial_skin_b_min * factor
        self.skin_b_max = self.initial_skin_b_max * factor + 255 * (1 - factor)
        
        # Blue colors (sky, water, etc.)
        self.blue_b_min = self.initial_blue_b_min * factor
        self.blue_bg_ratio_min = self.initial_blue_bg_ratio_min * factor + 10 * (1 - factor)  # As factor approaches 0, this becomes so high that no blue will be masked
        self.blue_br_ratio_min = self.initial_blue_br_ratio_min * factor + 10 * (1 - factor)  # Same here
    
    def is_valid_crop(self, img_array, i, j, h, w):
        """Check if a proposed crop region contains too many black pixels."""
        crop = img_array[i:i+h, j:j+w]
        pixel_sum = crop.sum(axis=2)  # Sum of RGB values for each pixel
        black_pixels = (pixel_sum < self.black_threshold).sum()
        total_pixels = crop.shape[0] * crop.shape[1]
        
        # Return True if the percentage of black pixels is below the threshold
        return (black_pixels / total_pixels) <= self.max_black_percentage
    
    def get_valid_crop_params(self, img, img_array, scale=(0.8, 1.0), ratio=(0.75, 1.33), max_attempts=10):
        """Get valid crop parameters that avoid regions with too many black pixels."""
        # Adjust crop scale based on epoch
        augment_factor = 1 - self.get_filtering_factor()  # Stronger augmentation as filtering reduces
        min_scale = scale[0] * (1 - 0.3 * augment_factor)  # Reduce minimum scale as training progresses
        
        for _ in range(max_attempts):
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(min_scale, scale[1]), ratio=ratio)
            if self.is_valid_crop(img_array, i, j, h, w):
                return i, j, h, w
        
        # If no valid crop found after max attempts, return a center crop
        w, h = img.size
        i = (h - min(h, w)) // 2
        j = (w - min(h, w)) // 2
        return i, j, min(h, w), min(h, w)
    
    def __call__(self, img):
        import numpy as np
        from PIL import ImageFilter, Image
        import os
        import uuid
        import random
        
        # Make a copy of the original image
        original_img = img.copy()
        
        # Resize the image to target size
        original_img = TF.resize(original_img, (self.img_size, self.img_size))
        
        # Get filtering factor - if it's 0, skip filtering entirely
        filtering_factor = self.get_filtering_factor()
        
        if filtering_factor > 0:
            # Create a blurred copy for masking
            blurred_img = original_img.filter(ImageFilter.GaussianBlur(radius=3))
            
            # Convert to numpy arrays for color manipulation
            np_orig = np.array(original_img)
            blurred = np.array(blurred_img)
            
            # Extract RGB channels from blurred image
            R = blurred[:,:,0].astype(float)
            G = blurred[:,:,1].astype(float)
            B = blurred[:,:,2].astype(float)
            
            # 1. Detect plant colors (what we want to keep)
            
            # Green vegetation
            green_mask = (G >= self.green_g_min) & ((B + R) <= (G * self.green_br_max_ratio))
            
            # Brown soil/wood
            brown_mask = ((R >= self.brown_r_min) & 
                        (G >= self.brown_g_min) & 
                        (B <= self.brown_b_max) &
                        (R > G) & (G > B))
            
            # Grey-brown vegetation (more desaturated browns found in some plants)
            grey_brown_mask = ((R >= self.grey_brown_r_min) &
                            (G >= self.grey_brown_g_min) &
                            (B >= self.grey_brown_b_min) &
                            (np.abs(R - G) <= self.grey_brown_max_diff) &
                            (np.abs(R - B) <= self.grey_brown_max_diff) &
                            (np.abs(G - B) <= self.grey_brown_max_diff))
            
            # Olive/yellow-green vegetation
            olive_mask = ((R >= self.olive_r_min) &
                        (G >= self.olive_g_min) &
                        (np.abs(R - G) <= self.olive_r_g_diff_max) &
                        (B <= self.olive_b_max))
            
            # 2. Detect colors to mask out
            
            # Skin tones
            skin_mask = ((R >= self.skin_r_min) & (R <= self.skin_r_max) &
                        (G >= self.skin_g_min) & (G <= self.skin_g_max) &
                        (B >= self.skin_b_min) & (B <= self.skin_b_max))
            
            # Blue colors (sky, water, objects)
            blue_mask = ((B >= self.blue_b_min) &
                        (B / (G + 1e-6) >= self.blue_bg_ratio_min) &
                        (B / (R + 1e-6) >= self.blue_br_ratio_min))
            
            # 3. Combine masks
            
            # Vegetation mask (what we want to keep)
            vegetation_mask = green_mask | brown_mask | grey_brown_mask | olive_mask
            
            # Combined mask for colors to remove
            remove_mask = skin_mask | blue_mask
            
            # Final mask: keep vegetation but remove skin and blue even if they match vegetation criteria
            keep_mask = vegetation_mask & ~remove_mask
            
            # Apply mask to the original (non-blurred) image
            masked_img = np_orig.copy()
            masked_img[~keep_mask] = [0, 0, 0]  # Set non-vegetation to flat black
            
            # Convert back to PIL Image
            masked_pil = Image.fromarray(masked_img.astype(np.uint8))
            
            # Save the masked image to output directory if we're doing significant filtering
            if filtering_factor > 0.1:
                filename = f"masked_epoch{self.current_epoch}_{uuid.uuid4().hex}.jpg"
                masked_pil.save(os.path.join(self.output_dir, filename))
            
            # Continue with augmentation pipeline on the masked image
            img = masked_pil
            masked_array = masked_img  # Keep numpy array for crop validation
        else:
            # Skip masking if filtering factor is 0
            img = original_img
            masked_array = np.array(original_img)
        
        # Get augmentation probability that increases over epochs
        aug_prob = self.get_augmentation_probability()
        
        # Apply random crop and resize while avoiding black regions
        if random.random() < aug_prob:
            i, j, h, w = self.get_valid_crop_params(img, masked_array)
            img = TF.resized_crop(img, i, j, h, w, (self.img_size, self.img_size))
        else:
            img = TF.resize(img, (self.img_size, self.img_size))
            
        # Apply random horizontal flip
        if random.random() < aug_prob:
            img = TF.hflip(img)
            
        # Apply random vertical flip
        if random.random() < aug_prob:
            img = TF.vflip(img)
            
        # Apply random rotation - stronger with higher epochs
        if random.random() < aug_prob:
            # Rotation angle increases with epochs
            max_angle = 15 + 15 * (1 - self.get_filtering_factor())  # 15° early, up to 30° later
            angle = random.uniform(-max_angle, max_angle)
            img = TF.rotate(img, angle)
            
        # Apply random color jitter - stronger with higher epochs
        if random.random() < aug_prob:
            # Color jitter intensity increases with epochs
            jitter_factor = 0.1 + 0.1 * (1 - self.get_filtering_factor())
            brightness = random.uniform(1 - jitter_factor, 1 + jitter_factor)
            contrast = random.uniform(1 - jitter_factor, 1 + jitter_factor)
            saturation = random.uniform(1 - jitter_factor, 1 + jitter_factor)
            hue = random.uniform(-jitter_factor/2, jitter_factor/2)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
            img = TF.adjust_hue(img, hue)
            
        # Convert to tensor and normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        # Apply random erasing with probability that increases with epochs
        if random.random() < aug_prob:
            img = self.random_erasing(img)
            
        return img



# Basic building blocks
class ConvBnSiLU(nn.Module):
    """Convolutional + BatchNorm + SiLU activation (CBS block)"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Bottleneck block as shown in Figure 7"""
    def __init__(self, in_channels, out_channels, k=3):
        super().__init__()
        self.cbs1 = ConvBnSiLU(in_channels, out_channels, kernel_size=1)
        self.cbs2 = ConvBnSiLU(out_channels, out_channels, kernel_size=k)
        
        # Add a projection shortcut if dimensions don't match
        self.has_proj = in_channels != out_channels
        if self.has_proj:
            self.proj = ConvBnSiLU(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        if self.has_proj:
            return self.proj(x) + self.cbs2(self.cbs1(x))
        else:
            return x + self.cbs2(self.cbs1(x))

class C2f(nn.Module):
    """C2f module as shown in Figure 6"""
    def __init__(self, in_channels, out_channels, n=3):
        super().__init__()
        self.n = n
        self.in_conv = ConvBnSiLU(in_channels, out_channels, kernel_size=1)
        
        # Calculate exact chunk sizes to handle uneven division
        self.chunk_sizes = []
        for i in range(n):
            size = out_channels // n
            if i < out_channels % n:  # Distribute remainder
                size += 1
            self.chunk_sizes.append(size)
        
        # Calculate total channels after concatenation
        # First chunk + all processed chunks
        concat_channels = self.chunk_sizes[0] + sum(self.chunk_sizes)
        self.out_conv = ConvBnSiLU(concat_channels, out_channels, kernel_size=1)
        
        # Create bottlenecks with exact sizes
        self.bottlenecks = nn.ModuleList([
            Bottleneck(size, size)
            for size in self.chunk_sizes
        ])
    
    def forward(self, x):
        x = self.in_conv(x)
        
        # Split tensor into chunks with the exact pre-calculated sizes
        xs = []
        start_idx = 0
        for size in self.chunk_sizes:
            xs.append(x[:, start_idx:start_idx+size, :, :])
            start_idx += size
        
        # Process each chunk through its bottleneck
        processed = [bottleneck(part) for bottleneck, part in zip(self.bottlenecks, xs)]
        
        # Concatenate first chunk with all processed chunks
        out = torch.cat([xs[0]] + processed, dim=1)
        
        return self.out_conv(out)

class SPP(nn.Module):
    """Spatial Pyramid Pooling module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBnSiLU(in_channels, in_channels // 2, kernel_size=1)
        self.cv2 = ConvBnSiLU(in_channels // 2 * 4, out_channels, kernel_size=1)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in (5, 9, 13)
        ])
    
    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention module for the C3TR block"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """MLP module for the Transformer block"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer Block for C3TR"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False, 
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            drop=drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class C3TR(nn.Module):
    """C3TR module that combines CNN and Transformer"""
    def __init__(self, in_channels, out_channels, num_heads=8, depth=1):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        self.transformer_dim = out_channels
        self.transformer = nn.ModuleList([
            TransformerBlock(self.transformer_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        
        self.conv_final = ConvBnSiLU(out_channels, out_channels, kernel_size=3)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv_in(x)
        
        # Reshape for transformer: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Apply transformer blocks
        for block in self.transformer:
            x_flat = block(x_flat)
        
        # Reshape back to [B, C, H, W]
        x_reshaped = x_flat.transpose(1, 2).view(B, C, H, W)
        
        # Final convolution
        x = self.conv_out(x_reshaped)
        x = self.conv_final(x)
        
        return x

class SC3T(nn.Module):
    """SC3T module combining SPP and C3TR as shown in Figure 8"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spp = SPP(in_channels, out_channels)
        self.c3tr = C3TR(out_channels, out_channels)
        self.conv = ConvBnSiLU(out_channels, out_channels, kernel_size=3)
    
    def forward(self, x):
        x = self.spp(x)
        x = self.c3tr(x)
        x = self.conv(x)
        return x

class FocusModule(nn.Module):
    """Focus module as shown in Figure 11"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBnSiLU(in_channels * 4, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Create 4 slices: top-left, top-right, bottom-left, bottom-right
        slices = []
        
        # Top-left
        slices.append(x[:, :, 0::2, 0::2])
        # Top-right
        slices.append(x[:, :, 0::2, 1::2])
        # Bottom-left
        slices.append(x[:, :, 1::2, 0::2])
        # Bottom-right
        slices.append(x[:, :, 1::2, 1::2])
        
        # Concatenate along channel dimension
        x = torch.cat(slices, dim=1)
        
        # Apply convolution
        x = self.conv(x)
        
        return x

class YOLOv8TransformerClassifier(nn.Module):
    """Complete YOLOv8-Transformer model for plant disease classification"""
    def __init__(self, in_channels=3, num_classes=14):
        super().__init__()
        
        # Base channel sizes
        channels = [64, 128, 256, 512, 1024]
        
        # Initial focus module
        self.focus = FocusModule(in_channels, channels[0])
        
        # Backbone
        self.backbone = nn.ModuleList([
            # Down-sampling blocks
            nn.Sequential(
                ConvBnSiLU(channels[0], channels[1], kernel_size=3, stride=2),
                C2f(channels[1], channels[1])
            ),
            nn.Sequential(
                ConvBnSiLU(channels[1], channels[2], kernel_size=3, stride=2),
                C2f(channels[2], channels[2])
            ),
            nn.Sequential(
                ConvBnSiLU(channels[2], channels[3], kernel_size=3, stride=2),
                C2f(channels[3], channels[3])
            ),
            nn.Sequential(
                ConvBnSiLU(channels[3], channels[4], kernel_size=3, stride=2),
                C2f(channels[4], channels[4])
            )
        ])
        
        # SC3T module in the final layer
        self.sc3t = SC3T(channels[4], channels[4])
        
        # Head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[4], num_classes)
        
    def forward(self, x):
        # Initial processing
        x = self.focus(x)
        
        # Backbone
        for block in self.backbone:
            x = block(x)
        
        # SC3T module
        x = self.sc3t(x)
        
        # Classification head
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# Data Preparation
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths"""
    def __getitem__(self, index):
        # Get the original tuple (image, label)
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        # Get the path to the image file
        path = self.imgs[index][0]
        return img, label, path

def get_data_loaders(data_dir, img_size, batch_size, num_workers, current_epoch=0, 
                     max_filtering_epochs=30, total_epochs=100, output_dir="masked_images"):
    """Create and return train and test data loaders with epoch-aware augmentation"""
    
    # Define the custom augmentation with epoch information
    custom_aug = CustomAugmentation(
        img_size=img_size, 
        prob=0.5,
        output_dir=output_dir,
        current_epoch=current_epoch,
        max_filtering_epochs=max_filtering_epochs,
        total_epochs=total_epochs
    )
    
    # Define transforms for training and testing
    train_transform = transforms.Compose([
        custom_aug,
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create train and test datasets
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolderWithPaths(root=test_dir, transform=test_transform)
    
    # Check if classes match our expected classes
    print(f"Found classes: {train_dataset.classes}")
    assert len(train_dataset.classes) == len(CLASSES), f"Expected {len(CLASSES)} classes, but found {len(train_dataset.classes)}"
    
    # Compute class weights for handling imbalanced data
    class_weights = None
    if Config.class_weights:
        # Count samples per class
        class_counts = [0] * len(train_dataset.classes)
        for _, label in train_dataset.samples:
            class_counts[label] += 1
        
        # Compute weights inversely proportional to class frequencies
        total_samples = sum(class_counts)
        class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])
        print(f"Class weights: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print dataset statistics
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # Return loaders, weights, and the custom augmentation object for epoch updates
    return train_loader, test_loader, class_weights, custom_aug


# Train function (same as before)
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_start_time = time.time()
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        # Handle different return formats (with or without paths)
        if len(batch) == 3:
            inputs, labels, _ = batch  # Ignore paths if present
        else:
            inputs, labels = batch
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })
    
    epoch_time = time.time() - epoch_start_time
    
    # Return epoch statistics
    return running_loss / total, 100. * correct / total, epoch_time

def validate(model, val_loader, criterion, device):
    """Validate the model on the validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Storage for prediction stats
    all_predictions = []
    all_targets = []
    
    val_start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Handle both cases: with or without paths
            if len(batch) == 3:
                inputs, labels, _ = batch  # Unpack and ignore the paths
            else:
                inputs, labels = batch
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions and targets for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate validation accuracy and loss
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    val_time = time.time() - val_start_time
    
    # Return validation statistics
    return val_loss, val_acc, all_predictions, all_targets, val_time

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model performance with detailed metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_image_paths = []
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Handle both cases: with or without paths
            if len(batch) == 3:
                inputs, labels, paths = batch
                all_image_paths.extend(paths)
            else:
                inputs, labels = batch
                # Create dummy paths if real paths aren't available
                all_image_paths.extend(["unknown_path"] * labels.size(0))
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Store predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Get detailed classification report
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Identify misclassified samples
    misclassified_indices = np.where(y_true != y_pred)[0]
    misclassified = [
        {
            'path': all_image_paths[i],
            'true_label': class_names[y_true[i]],
            'pred_label': class_names[y_pred[i]]
        }
        for i in misclassified_indices
    ]
    
    # Print results
    print(f"\nTest Accuracy: {100 * np.mean(y_true == y_pred):.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    eval_time = time.time() - eval_start_time
    
    return {
        'confusion_matrix': cm,
        'classification_report': class_report,
        'misclassified': misclassified,
        'accuracy': np.mean(y_true == y_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'eval_time': eval_time
    }

def plot_results(results, class_names, log_dir):
    """Plot and save evaluation results"""
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot confusion matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix')
    
    # Plot class-wise precision, recall, and F1
    metrics = {}
    for cls in class_names:
        metrics[cls] = [
            results['classification_report'][cls]['precision'],
            results['classification_report'][cls]['recall'],
            results['classification_report'][cls]['f1-score']
        ]
    
    metrics_df = pd.DataFrame(metrics, index=['Precision', 'Recall', 'F1-Score']).T
    metrics_df.plot(kind='bar', ax=ax2)
    ax2.set_title('Class-wise Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='lower right')
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'evaluation_results.png'))
    plt.close()
    
    # Return for displaying
    return fig

def save_misclassified_examples(results, log_dir, num_examples=10):
    """Save a sample of misclassified images for analysis"""
    misclassified = results['misclassified']
    
    if len(misclassified) == 0:
        print("No misclassified examples to show!")
        return None
    
    # Randomly sample misclassified examples
    samples = random.sample(misclassified, min(num_examples, len(misclassified)))
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6)) if num_examples >= 10 else plt.subplots(
        1, len(samples), figsize=(15, 3))
    axes = axes.flatten()
    
    # Plot each sample
    for i, sample in enumerate(samples):
        if i >= len(axes):
            break
            
        # Load image
        img = Image.open(sample['path'])
        axes[i].imshow(img)
        axes[i].set_title(f"True: {sample['true_label']}\nPred: {sample['pred_label']}")
        axes[i].axis('off')
    
    # Hide unused subplots
    for j in range(len(samples), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'misclassified_examples.png'))
    plt.close()
    
    # Return for displaying
    return fig

def predict_single_image(model, image_path, class_names, device, img_size=150):
    """Predict the class of a single image"""
    # Load and transform the image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
    
    # Get predicted class and confidence
    pred_class = class_names[predicted.item()]
    confidence = probs[0][predicted.item()].item()
    
    # Get top-3 predictions
    top3_values, top3_indices = torch.topk(probs, 3, dim=1)
    top3_predictions = [
        (class_names[idx.item()], val.item())
        for idx, val in zip(top3_indices[0], top3_values[0])
    ]
    
    return {
        'predicted_class': pred_class,
        'confidence': confidence,
        'top3_predictions': top3_predictions
    }

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_loss, 
                    train_acc, train_loss, filename, metadata=None):
    """Save model checkpoint with all training information"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'metadata': metadata or {}
    }, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """Load model checkpoint and return training information"""
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return None
    
    checkpoint = torch.load(filename, map_location=Config.device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optionally load optimizer and scheduler states
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_acc']:.2f}%")
    
    return checkpoint

def save_training_history(history, filename):
    """Save training history to JSON file"""
    with open(filename, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {filename}")

def load_training_history(filename):
    """Load training history from JSON file"""
    if not os.path.exists(filename):
        return []
    
    with open(filename, 'r') as f:
        history = json.load(f)
    
    return history

def plot_training_history(history, filename):
    """Plot training history and save to file"""
    epochs = [entry['epoch'] for entry in history]
    train_loss = [entry['train_loss'] for entry in history]
    val_loss = [entry['val_loss'] for entry in history]
    train_acc = [entry['train_acc'] for entry in history]
    val_acc = [entry['val_acc'] for entry in history]
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    # Return for displaying
    return fig

def train_and_evaluate(resume_training=None, max_filtering_epochs=None, masking_output_dir=None):
    """Full training and evaluation pipeline with progressive vegetation masking"""
    
    # Use config values if not specified
    if max_filtering_epochs is None:
        max_filtering_epochs = Config.max_filtering_epochs
    
    if masking_output_dir is None:
        masking_output_dir = Config.masking_output_dir
    
    # Create mask output directory
    os.makedirs(masking_output_dir, exist_ok=True)
    
    # Initialize training variables
    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0
    history = []
    training_start_time = time.time()
    
    # Path for checkpoints
    best_model_path = os.path.join(Config.save_dir, 'best_model.pth')
    latest_model_path = os.path.join(Config.save_dir, 'latest_model.pth')
    history_path = os.path.join(Config.log_dir, 'training_history.json')
    
    # Try to resume training if specified
    if resume_training is None:
        resume_training = Config.resume_training
    
    # Initialize model
    model = YOLOv8TransformerClassifier(in_channels=3, num_classes=len(CLASSES))
    model = model.to(Config.device)
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.learning_rate, 
        weight_decay=Config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Try to load checkpoint if resuming
    if resume_training:
        # Try to load best model first, then latest
        checkpoint = load_checkpoint(best_model_path, model, optimizer, scheduler)
        if checkpoint is None:
            checkpoint = load_checkpoint(latest_model_path, model, optimizer, scheduler)
        
        if checkpoint:
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"Resuming training from epoch {start_epoch+1} with best validation accuracy: {best_val_acc:.2f}%")
            
            # Load training history
            history = load_training_history(history_path)
    
    # Setup data loaders with the current epoch for progressive masking
    train_loader, test_loader, class_weights, custom_aug = get_data_loaders(
        Config.data_dir, 
        Config.img_size, 
        Config.batch_size, 
        Config.num_workers,
        current_epoch=start_epoch,  # Start at the appropriate epoch
        max_filtering_epochs=max_filtering_epochs,
        total_epochs=Config.num_epochs,
        output_dir=masking_output_dir
    )
    
    # Define loss function
    if Config.class_weights and class_weights is not None:
        class_weights = class_weights.to(Config.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted loss function")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create metadata dictionary for tracking
    metadata = {
        'train_start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(Config.device),
        'data_dir': Config.data_dir,
        'img_size': Config.img_size,
        'batch_size': Config.batch_size,
        'initial_lr': Config.learning_rate,
        'max_filtering_epochs': max_filtering_epochs,
        'masking_output_dir': masking_output_dir
    }
    
    print(f"Starting training on {Config.device}")
    print(f"Training from epoch {start_epoch+1} to {Config.num_epochs}")
    print(f"Vegetation filtering will decrease over first {max_filtering_epochs} epochs")
    
    # Training loop
    for epoch in range(start_epoch + 1, Config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{Config.num_epochs}")
        
        # Update custom augmentation with current epoch
        custom_aug.update_epoch(epoch - 1)  # Epoch is 1-indexed, update_epoch expects 0-indexed
        
        # Print current filtering/augmentation status
        filtering_factor = custom_aug.get_filtering_factor()
        aug_prob = custom_aug.get_augmentation_probability()
        print(f"Filtering strength: {filtering_factor:.2f}, Augmentation probability: {aug_prob:.2f}")
        
        # Train
        train_loss, train_acc, epoch_time = train_one_epoch(model, train_loader, criterion, optimizer, Config.device)
        
        # Validate
        val_loss, val_acc, predictions, targets, val_time = validate(model, test_loader, criterion, Config.device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {epoch_time:.2f}s")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {val_time:.2f}s")
        
        # Save epoch statistics
        epoch_stats = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
            'val_time': val_time,
            'filtering_factor': filtering_factor,
            'aug_prob': aug_prob,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        history.append(epoch_stats)
        
        # Save latest model
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_acc, val_loss, 
            train_acc, train_loss, latest_model_path, 
            metadata={
                **metadata,
                'history': epoch_stats,
                'training_time_so_far': time.time() - training_start_time
            }
        )
        
        # Save training history
        save_training_history(history, history_path)
        
        # Plot training history
        plot_training_history(history, os.path.join(Config.log_dir, 'training_history.png'))
        
        # Periodic checkpoints
        if epoch % Config.checkpoint_freq == 0:
            checkpoint_path = os.path.join(Config.save_dir, f'model_epoch_{epoch}.pth')
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc, val_loss, 
                train_acc, train_loss, checkpoint_path
            )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save the model
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc, val_loss, 
                train_acc, train_loss, best_model_path,
                metadata={
                    **metadata,
                    'best_epoch': epoch,
                    'training_time': time.time() - training_start_time
                }
            )
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= Config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break
    
    # Total training time
    total_training_time = time.time() - training_start_time
    print(f"Training completed in {total_training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Update metadata
    metadata.update({
        'best_val_acc': best_val_acc,
        'total_training_time': total_training_time,
        'train_end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Load best model for evaluation
    checkpoint = load_checkpoint(best_model_path, model)
    
    # Final evaluation
    print("Performing final evaluation...")
    results = evaluate_model(model, test_loader, Config.device, CLASSES)
    
    # Plot results
    try:
        plot_results(results, CLASSES, Config.log_dir)
        save_misclassified_examples(results, Config.log_dir)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Save evaluation results
    evaluation_path = os.path.join(Config.log_dir, 'evaluation_results.json')
    with open(evaluation_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for k, v in results.items():
            if k == 'confusion_matrix':
                serializable_results[k] = v.tolist()
            elif isinstance(v, np.ndarray):
                serializable_results[k] = v.tolist()
            else:
                serializable_results[k] = v
                
        # Add metadata
        serializable_results['metadata'] = metadata
                
        json.dump(serializable_results, f, indent=4)
    
    print(f"Evaluation results saved to {evaluation_path}")
    
    return model, results

# Inference function for deployment
def prepare_inference_model(model_path=None):
    """Load trained model for inference"""
    if model_path is None:
        model_path = os.path.join(Config.save_dir, 'best_model.pth')
        
    # Initialize model
    model = YOLOv8TransformerClassifier(in_channels=3, num_classes=len(CLASSES))
    
    # Load model weights
    checkpoint = load_checkpoint(model_path, model)
    if checkpoint is None:
        print(f"No model found at {model_path}. Please train the model first.")
        return None
        
    model = model.to(Config.device)
    model.eval()
    
    return model

# Batch inference for multiple images
def batch_inference(model, image_dir, class_names, device, img_size=150):
    """Run inference on all images in a directory"""
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [
        os.path.join(root, file) 
        for root, _, files in os.walk(image_dir) 
        for file in files 
        if os.path.splitext(file.lower())[1] in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"Found {len(image_files)} images for inference")
    results = []
    
    # Process each image
    inference_start_time = time.time()
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            prediction = predict_single_image(model, image_path, class_names, device, img_size)
            results.append({
                'image_path': image_path,
                'prediction': prediction['predicted_class'],
                'confidence': prediction['confidence'],
                'top3': prediction['top3_predictions']
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    inference_time = time.time() - inference_start_time
    
    # Save results to CSV
    try:
        import pandas as pd
        results_df = pd.DataFrame(results)
        output_path = os.path.join(os.path.dirname(image_dir), f'inference_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    except ImportError:
        print("Pandas not available. Saving results as JSON.")
        output_path = os.path.join(os.path.dirname(image_dir), f'inference_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")
    
    print(f"Inference completed in {inference_time:.2f} seconds for {len(results)} images")
    print(f"Average inference time per image: {inference_time/len(results):.4f} seconds")
    
    return results

def main():
    """Main function to handle training and inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plant Disease Classification')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'],
                        help='Operation mode: train or infer')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for inference')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory of images for inference')
    # Add arguments for vegetation filtering control
    parser.add_argument('--max_filtering_epochs', type=int, default=Config.max_filtering_epochs, 
                        help='Number of epochs over which to phase out vegetation filtering')
    parser.add_argument('--masking_output_dir', type=str, default=Config.masking_output_dir,
                        help='Directory to save masked images')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting plant disease classification training...")
        # Override Config.resume_training with command line argument
        resume_training = args.resume if args.resume is not None else Config.resume_training
        model, results = train_and_evaluate(
            resume_training=resume_training,
            max_filtering_epochs=args.max_filtering_epochs,
            masking_output_dir=args.masking_output_dir
        )
        print("Training completed!")
    
    elif args.mode == 'infer':
        print("Starting inference mode...")
        model = prepare_inference_model(args.model_path)
        
        if model is None:
            print("Model loading failed. Exiting.")
            return
        
        if args.image_dir:
            # Batch inference
            batch_inference(model, args.image_dir, CLASSES, Config.device, Config.img_size)
        else:
            # Example inference
            print("\nExample inference:")
            test_image_path = os.path.join(Config.data_dir, 'test', CLASSES[0], 
                              os.listdir(os.path.join(Config.data_dir, 'test', CLASSES[0]))[0])
            
            prediction = predict_single_image(model, test_image_path, CLASSES, Config.device)
            print(f"Test image: {test_image_path}")
            print(f"Predicted class: {prediction['predicted_class']}")
            print(f"Confidence: {prediction['confidence']:.4f}")
            print(f"Top 3 predictions: {prediction['top3_predictions']}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()