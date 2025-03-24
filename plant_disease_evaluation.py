import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import argparse
from tqdm import tqdm

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
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'models'
    output_dir = 'evaluation_results'
    
# Make sure output directory exists
os.makedirs(Config.output_dir, exist_ok=True)

# =============================================================================
# Model Definition (from original code)
# =============================================================================

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
    """Bottleneck block"""
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
    """C2f module"""
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
    """SC3T module combining SPP and C3TR"""
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
    """Focus module"""
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

# =============================================================================
# Data Loading Functions
# =============================================================================

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths"""
    def __getitem__(self, index):
        # Get the original tuple (image, label)
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        # Get the path to the image file
        path = self.imgs[index][0]
        return img, label, path

def get_test_loader(data_dir, img_size, batch_size, num_workers):
    """Create and return test data loader"""
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = ImageFolderWithPaths(root=test_dir, transform=test_transform)
    
    # Verify classes match expected classes
    print(f"Found classes: {test_dataset.classes}")
    assert len(test_dataset.classes) == len(CLASSES), f"Expected {len(CLASSES)} classes, but found {len(test_dataset.classes)}"
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    return test_loader

# =============================================================================
# Model Loading and Evaluation Functions
# =============================================================================

def load_model(model_path, device):
    """Load a trained model from checkpoint"""
    # Initialize model
    model = YOLOv8TransformerClassifier(in_channels=3, num_classes=len(CLASSES))
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
        print(f"Model was trained for {checkpoint['epoch']} epochs")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A')}%")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model performance with detailed metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_image_paths = []
    all_confidences = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, labels, paths = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, dim=1)
            
            # Store predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_image_paths.extend(paths)
            all_confidences.extend(confidences.cpu().numpy())
    
    # Convert lists to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    confidences = np.array(all_confidences)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Get detailed classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Identify misclassified samples
    misclassified_indices = np.where(y_true != y_pred)[0]
    misclassified = [
        {
            'path': all_image_paths[i],
            'true_label': class_names[y_true[i]],
            'pred_label': class_names[y_pred[i]],
            'confidence': confidences[i]
        }
        for i in misclassified_indices
    ]
    
    # Identify most commonly confused pairs
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true': class_names[i],
                    'predicted': class_names[j],
                    'count': cm[i, j],
                    'percentage': 100 * cm[i, j] / np.sum(cm[i, :])
                })
    
    # Sort by count
    confusion_pairs = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    
    print(f"\nTest Accuracy: {100 * accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'misclassified': misclassified,
        'confusion_pairs': confusion_pairs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'confidences': confidences
    }

# =============================================================================
# Improved Visualization Functions
# =============================================================================

def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot an improved confusion matrix with better readability"""
    # Calculate percentage confusion matrix (normalize by row)
    cm_percentage = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure for the confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot raw counts confusion matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=class_names, yticklabels=class_names, ax=ax1,
        annot_kws={"size": 10}
    )
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14)
    ax1.xaxis.set_ticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax1.yaxis.set_ticklabels(class_names, rotation=0, fontsize=10)
    
    # Plot percentage confusion matrix
    sns.heatmap(
        cm_percentage, annot=True, fmt='.1f', cmap='YlGnBu', 
        xticklabels=class_names, yticklabels=class_names, ax=ax2,
        annot_kws={"size": 10}
    )
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_title('Confusion Matrix (Percentage by Row)', fontsize=14)
    ax2.xaxis.set_ticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax2.yaxis.set_ticklabels(class_names, rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a more readable large format confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 12}
    )
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix (Counts)', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_large.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_metrics(report, output_dir):
    """Plot class-wise precision, recall, and F1-score with improved readability"""
    # Extract metrics for each class
    classes = []
    precision = []
    recall = []
    f1_score = []
    support = []
    
    for cls, metrics in report.items():
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(cls)
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1_score.append(metrics['f1-score'])
            support.append(metrics['support'])
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Support': support
    })
    
    # Sort by F1-score
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create grouped bar chart
    x = np.arange(len(metrics_df))
    width = 0.25
    
    ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#5DA5DA')
    ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#FAA43A')
    ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#60BD68')
    
    # Customize plot
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Class-wise Performance Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Class'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1)
    
    # Add a grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add class support as text on top of bars
    for i, support in enumerate(metrics_df['Support']):
        ax.text(i, 0.05, f'n={support}', ha='center', fontsize=8, color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics as CSV
    metrics_df.to_csv(os.path.join(output_dir, 'class_metrics.csv'), index=False)
    
    return metrics_df

def plot_confusion_pairs(confusion_pairs, output_dir, top_n=15):
    """Plot the most commonly confused class pairs"""
    if not confusion_pairs:
        print("No confusion pairs to plot.")
        return
    
    # Take top N most confused pairs
    top_pairs = confusion_pairs[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    pair_names = [f"{p['true']} → {p['predicted']}" for p in top_pairs]
    counts = [p['count'] for p in top_pairs]
    percentages = [p['percentage'] for p in top_pairs]
    
    # Sort by count
    sorted_indices = np.argsort(counts)
    pair_names = [pair_names[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    percentages = [percentages[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    bars = ax.barh(pair_names, counts, color='#5DA5DA')
    
    # Add count and percentage annotations
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 0.5, bar.get_y() + bar.get_height()/2,
            f"{counts[i]} ({percentages[i]:.1f}%)",
            va='center', fontsize=10
        )
    
    ax.set_xlabel('Number of Instances', fontsize=12)
    ax.set_title('Top Most Confused Class Pairs (True → Predicted)', fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_pairs.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_misclassification_examples(results, output_dir, num_examples=15):
    """Plot examples of misclassified images with improved layout"""
    misclassified = results['misclassified']
    
    if len(misclassified) == 0:
        print("No misclassified examples to show!")
        return
    
    # Randomly sample misclassified examples
    samples = random.sample(misclassified, min(num_examples, len(misclassified)))
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(len(samples))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    # Flatten axes for easy indexing
    axes = axes.flatten()
    
    # Plot each sample
    for i, sample in enumerate(samples):
        if i >= len(axes):
            break
            
        try:
            # Load image
            img = Image.open(sample['path'])
            axes[i].imshow(img)
            axes[i].set_title(
                f"True: {sample['true_label']}\nPred: {sample['pred_label']}\nConf: {sample['confidence']:.2f}",
                fontsize=8
            )
            axes[i].axis('off')
        except Exception as e:
            print(f"Error displaying image {sample['path']}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading image", ha='center', va='center')
            axes[i].axis('off')
    
    # Hide unused subplots
    for j in range(len(samples), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'misclassified_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_distribution(results, output_dir):
    """Plot the distribution of confidence scores for correct and incorrect predictions"""
    y_true = results['y_true']
    y_pred = results['y_pred']
    confidences = results['confidences']
    
    # Separate confidences for correct and incorrect predictions
    correct_mask = y_true == y_pred
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot histograms
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    
    ax1.hist(correct_confidences, bins=bins, alpha=0.7, color='green', label=f'Correct (n={len(correct_confidences)})')
    ax1.hist(incorrect_confidences, bins=bins, alpha=0.7, color='red', label=f'Incorrect (n={len(incorrect_confidences)})')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Confidence Distribution', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot KDE
    try:
        sns.kdeplot(correct_confidences, ax=ax2, color='green', fill=True, alpha=0.3, label=f'Correct (n={len(correct_confidences)})')
        if len(incorrect_confidences) > 0:
            sns.kdeplot(incorrect_confidences, ax=ax2, color='red', fill=True, alpha=0.3, label=f'Incorrect (n={len(incorrect_confidences)})')
        ax2.set_xlabel('Confidence Score', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Confidence Density', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        print(f"Error plotting KDE: {e}")
        ax2.text(0.5, 0.5, "KDE plot unavailable", ha='center', va='center')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a table of confidence statistics
    confidence_stats = {
        'Prediction': ['Correct', 'Incorrect', 'Overall'],
        'Count': [len(correct_confidences), len(incorrect_confidences), len(confidences)],
        'Min': [correct_confidences.min() if len(correct_confidences) > 0 else np.nan,
                incorrect_confidences.min() if len(incorrect_confidences) > 0 else np.nan,
                confidences.min()],
        'Max': [correct_confidences.max() if len(correct_confidences) > 0 else np.nan,
                incorrect_confidences.max() if len(incorrect_confidences) > 0 else np.nan,
                confidences.max()],
        'Mean': [correct_confidences.mean() if len(correct_confidences) > 0 else np.nan,
                 incorrect_confidences.mean() if len(incorrect_confidences) > 0 else np.nan,
                 confidences.mean()],
        'Median': [np.median(correct_confidences) if len(correct_confidences) > 0 else np.nan,
                  np.median(incorrect_confidences) if len(incorrect_confidences) > 0 else np.nan,
                  np.median(confidences)]
    }
    
    # Save as CSV
    pd.DataFrame(confidence_stats).to_csv(os.path.join(output_dir, 'confidence_stats.csv'), index=False)

def generate_overall_summary(results, class_names, output_dir):
    """Generate an overall summary of model performance"""
    # Get overall metrics
    accuracy = results['accuracy']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1']
    
    # Get class-wise metrics
    report = results['classification_report']
    
    # Create summary dictionary
    summary = {
        'overall_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'class_counts': {
            cls: report[cls]['support'] for cls in class_names
        },
        'best_performing_classes': [],
        'worst_performing_classes': [],
        'confused_pairs': results['confusion_pairs'][:5] if results['confusion_pairs'] else []
    }
    
    # Find best and worst performing classes
    class_f1 = [(cls, report[cls]['f1-score']) for cls in class_names]
    class_f1.sort(key=lambda x: x[1], reverse=True)
    
    # Add top 3 best and worst classes
    summary['best_performing_classes'] = class_f1[:3]
    summary['worst_performing_classes'] = class_f1[-3:]
    
    # Save summary as JSON
    with open(os.path.join(output_dir, 'model_summary.json'), 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        serializable_summary = json.loads(json.dumps(summary, default=convert_to_serializable))
        json.dump(serializable_summary, f, indent=4)
    
    # Generate a text summary
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("=======================\n\n")
        
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Overall Precision: {precision:.4f}\n")
        f.write(f"Overall Recall: {recall:.4f}\n")
        f.write(f"Overall F1 Score: {f1:.4f}\n\n")
        
        f.write("Class Distribution:\n")
        for cls, count in summary['class_counts'].items():
            f.write(f"  {cls}: {count} samples\n")
        f.write("\n")
        
        f.write("Best Performing Classes (by F1 Score):\n")
        for cls, score in summary['best_performing_classes']:
            f.write(f"  {cls}: {score:.4f}\n")
        f.write("\n")
        
        f.write("Worst Performing Classes (by F1 Score):\n")
        for cls, score in summary['worst_performing_classes']:
            f.write(f"  {cls}: {score:.4f}\n")
        f.write("\n")
        
        f.write("Most Confused Class Pairs:\n")
        for pair in summary['confused_pairs']:
            f.write(f"  {pair['true']} misclassified as {pair['predicted']}: {pair['count']} times ({pair['percentage']:.1f}%)\n")
    
    print(f"Summary saved to {os.path.join(output_dir, 'model_summary.txt')}")
    
    return summary

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to load model and evaluate it"""
    parser = argparse.ArgumentParser(description='Plant Disease Classification Model Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default=Config.data_dir, help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default=Config.output_dir, help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=Config.batch_size, help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=Config.img_size, help='Image size for evaluation')
    parser.add_argument('--num_workers', type=int, default=Config.num_workers, help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Update Config with command line arguments
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.batch_size = args.batch_size
    Config.img_size = args.img_size
    Config.num_workers = args.num_workers
    
    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, Config.device)
    if model is None:
        return
    
    # Get test loader
    test_loader = get_test_loader(Config.data_dir, Config.img_size, Config.batch_size, Config.num_workers)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, Config.device, CLASSES)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], CLASSES, Config.output_dir)
    
    # Plot class metrics
    plot_class_metrics(results['classification_report'], Config.output_dir)
    
    # Plot confusion pairs
    plot_confusion_pairs(results['confusion_pairs'], Config.output_dir)
    
    # Plot misclassified examples
    plot_misclassification_examples(results, Config.output_dir)
    
    # Plot confidence distribution
    plot_confidence_distribution(results, Config.output_dir)
    
    # Generate overall summary
    generate_overall_summary(results, CLASSES, Config.output_dir)
    
    print(f"Evaluation complete. Results saved to {Config.output_dir}")

if __name__ == "__main__":
    main()
   # python plant_disease_evaluation.py --model_path B:/DeepLearning/Train_over_cleaning/models/best_model.pth --data_dir B:/DeepLearning/dataset