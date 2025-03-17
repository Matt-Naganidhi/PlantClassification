import os
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    data_dir = 'dataset'  # Change this to your dataset path
    img_size = 640
    batch_size = 16
    num_workers = 4
    num_epochs = 50
    learning_rate = 1e-3
    weight_decay = 5e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'models'
    log_dir = 'logs'
    early_stopping_patience = 10
    class_weights = False  # Set to True to use class weights

# Make sure save directories exist
for directory in [Config.save_dir, Config.log_dir]:
    os.makedirs(directory, exist_ok=True)

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
    
    def forward(self, x):
        return x + self.cbs2(self.cbs1(x))

class C2f(nn.Module):
    """C2f module as shown in Figure 6"""
    def __init__(self, in_channels, out_channels, n=3):
        super().__init__()
        self.in_conv = ConvBnSiLU(in_channels, out_channels, kernel_size=1)
        self.out_conv = ConvBnSiLU(out_channels, out_channels, kernel_size=1)
        
        # Create n parallel bottlenecks
        self.bottlenecks = nn.ModuleList([
            Bottleneck(out_channels // n, out_channels // n)
            for _ in range(n)
        ])
    
    def forward(self, x):
        x = self.in_conv(x)
        
        # Split the channels into n equal parts
        xs = torch.chunk(x, len(self.bottlenecks), dim=1)
        
        # Process each part through bottlenecks
        processed = [bottleneck(part) for bottleneck, part in zip(self.bottlenecks, xs)]
        
        # Concatenate results
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

def get_data_loaders(data_dir, img_size, batch_size, num_workers):
    """Create and return train and test data loaders from directory structure"""
    
    # Define transforms for training and testing
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    
    return train_loader, test_loader, class_weights

# Training Functions
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for inputs, labels in progress_bar:
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
    
    # Return epoch statistics
    return running_loss / total, 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model on the validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Storage for prediction stats
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
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
    
    # Return validation statistics
    return val_loss, val_acc, all_predictions, all_targets

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model performance with detailed metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_image_paths = []
    
    with torch.no_grad():
        for inputs, labels, paths in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Store predictions, targets, and image paths
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_image_paths.extend(paths)
    
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
    
    return {
        'confusion_matrix': cm,
        'classification_report': class_report,
        'misclassified': misclassified,
        'accuracy': np.mean(y_true == y_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_results(results, class_names):
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
    plt.savefig(os.path.join(Config.log_dir, 'evaluation_results.png'))
    plt.close()

def save_misclassified_examples(results, num_examples=10):
    """Save a sample of misclassified images for analysis"""
    misclassified = results['misclassified']
    
    if len(misclassified) == 0:
        print("No misclassified examples to show!")
        return
    
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
    plt.savefig(os.path.join(Config.log_dir, 'misclassified_examples.png'))
    plt.close()

def predict_single_image(model, image_path, class_names, device, img_size=640):
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

# Main training function
def train_and_evaluate():
    """Full training and evaluation pipeline"""
    # Setup data loaders
    train_loader, test_loader, class_weights = get_data_loaders(
        Config.data_dir, Config.img_size, Config.batch_size, Config.num_workers
    )
    
    # Initialize model
    model = YOLOv8TransformerClassifier(in_channels=3, num_classes=len(CLASSES))
    model = model.to(Config.device)
    
    # Define loss function
    if Config.class_weights and class_weights is not None:
        class_weights = class_weights.to(Config.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted loss function")
    else:
        criterion = nn.CrossEntropyLoss()
    
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
    
    # Training loop
    best_val_acc = 0.0
    best_model_weights = None
    patience_counter = 0
    
    print(f"Starting training on {Config.device}")
    for epoch in range(1, Config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{Config.num_epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.device)
        
        # Validate
        val_loss, val_acc, predictions, targets = validate(model, test_loader, criterion, Config.device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss
            }, os.path.join(Config.save_dir, 'best_model.pth'))
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= Config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for evaluation
    model.load_state_dict(best_model_weights)
    
    # Final evaluation
    results = evaluate_model(model, test_loader, Config.device, CLASSES)
    
    # Plot results
    try:
        import pandas as pd
        plot_results(results, CLASSES)
        save_misclassified_examples(results)
    except ImportError:
        print("Pandas or Matplotlib not available. Skipping visualization.")
    
    return model, results

# Inference function for deployment
def prepare_inference_model(model_path):
    """Load trained model for inference"""
    # Initialize model
    model = YOLOv8TransformerClassifier(in_channels=3, num_classes=len(CLASSES))
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=Config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.device)
    model.eval()
    
    return model

# Batch inference for multiple images
def batch_inference(model, image_dir, class_names, device, img_size=640):
    """Run inference on all images in a directory"""
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [
        os.path.join(root, file) 
        for root, _, files in os.walk(image_dir) 
        for file in files 
        if os.path.splitext(file.lower())[1] in image_extensions
    ]
    
    results = []
    
    # Process each image
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
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(image_dir), 'inference_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")
    return results

if __name__ == "__main__":
    # Run the training pipeline
    print("Starting plant disease classification training...")
    model, results = train_and_evaluate()
    print("Training completed!")
    
    # Example of using the trained model for inference
    print("\nExample inference:")
    model_path = os.path.join(Config.save_dir, 'best_model.pth')
    inference_model = prepare_inference_model(model_path)
    
    # Replace this with your own test image path
    test_image_path = os.path.join(Config.data_dir, 'test', CLASSES[0], os.listdir(os.path.join(Config.data_dir, 'test', CLASSES[0]))[0])
    
    prediction = predict_single_image(inference_model, test_image_path, CLASSES, Config.device)
    print(f"Test image: {test_image_path}")
    print(f"Predicted class: {prediction['predicted_class']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"Top 3 predictions: {prediction['top3_predictions']}")
    
    print("\nDone!")