import os
import sys
import json
import argparse
import pathlib
import platform
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Cross-Platform Compatibility Patch
# ──────────────────────────────────────────────────────────────────────────────
# Resolves NotImplementedError when instantiating PosixPath on Windows environments.
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

try:
    import timm
except ImportError:
    print("Error: The 'timm' package is required for loading backbones.")
    print("Please install it via: pip install timm")
    sys.exit(1)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Error: The 'albumentations' package is required for image preprocessing.")
    print("Please install it via: pip install albumentations")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Model Architectures
# ──────────────────────────────────────────────────────────────────────────────
class GCPModelEffNetB3(nn.Module):
    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        feat_dim = self.backbone.num_features
        self.shared_fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.kp_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        self.cls_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        shared = self.shared_fc(feats)
        kp = self.kp_head(shared)
        cls = self.cls_head(shared)
        return kp, cls

class GCPModelConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        feat_dim = self.backbone.num_features
        self.kp_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        kp = self.kp_head(feats)
        cls = self.cls_head(feats)
        return kp, cls

# ──────────────────────────────────────────────────────────────────────────────
# Test Dataset Class
# ──────────────────────────────────────────────────────────────────────────────
class GCPTestDataset(Dataset):
    def __init__(self, image_paths, test_dir, transform):
        self.image_paths = image_paths
        self.test_dir = pathlib.Path(test_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        abs_path = self.image_paths[idx]
        img = np.array(Image.open(abs_path).convert('RGB'))
        oh, ow = img.shape[:2]
        res = self.transform(image=img)
        
        # Calculate relative path matching original annotation patterns
        try:
            rel_path = str(abs_path.relative_to(self.test_dir))
        except ValueError:
            rel_path = abs_path.name
        
        # Replace backslashes on Windows to keep consistent relative paths
        rel_path = rel_path.replace("\\", "/")
        
        return res['image'], rel_path, ow, oh

# ──────────────────────────────────────────────────────────────────────────────
# Test-Time Augmentation (TTA) Prediction
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def tta_predict(model, imgs):
    model.eval()
    # 1. Base inference
    kp0, cls0 = model(imgs)
    
    # 2. Horizontal Flip TTA
    kp1, cls1 = model(imgs.flip(-1))
    kp1[:, 0] = 1.0 - kp1[:, 0]  # Invert x-coordinate
    
    # 3. Vertical Flip TTA
    kp2, cls2 = model(imgs.flip(-2))
    kp2[:, 1] = 1.0 - kp2[:, 1]  # Invert y-coordinate
    
    # Average coordinates and label probabilities
    kp_mean = (kp0 + kp1 + kp2) / 3.0
    cls_mean = (F.softmax(cls0, dim=1) + F.softmax(cls1, dim=1) + F.softmax(cls2, dim=1)) / 3.0
    
    return kp_mean, cls_mean

# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Inference script for Aerial GCP Pose Estimation")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to test dataset directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to best_model.pth checkpoint")
    parser.add_argument("--output", type=str, default="predictions.json", help="Path to output predictions.json file")
    parser.add_argument("--batch_size", type=str, default=16, help="Batch size for inference")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        sys.exit(1)

    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state']
    
    # Read config saved in checkpoint
    config = ckpt.get('cfg', {})
    classes = config.get('CLASSES', ['Cross', 'L-Shaped', 'Square'])
    num_classes = len(classes)
    model_input = config.get('MODEL_INPUT', 384)
    idx2class = ckpt.get('idx2class', {0: 'Cross', 1: 'L-Shaped', 2: 'Square'})
    
    # In case idx2class keys are saved as strings in JSON/checkpoints
    idx2class = {int(k): v for k, v in idx2class.items()}

    # Automatic Architecture Detection
    is_convnext = any("stages" in k for k in state_dict.keys())
    if is_convnext:
        print("Detected backbone architecture: ConvNeXt-Tiny")
        model = GCPModelConvNeXtTiny(num_classes=num_classes, pretrained=False).to(device)
    else:
        print("Detected backbone architecture: EfficientNet-B3")
        model = GCPModelEffNetB3(num_classes=num_classes, pretrained=False).to(device)

    # Load state dict
    try:
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print("Failed to load model weights directly. Attempting compatibility fixes.")
        # Try stripping backbone prefixes if necessary
        fixed_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                fixed_dict[k[7:]] = v
            else:
                fixed_dict[k] = v
        model.load_state_dict(fixed_dict)
        print("Model weights loaded successfully (with fixes).")

    # Set up test directories
    test_dir_path = args.test_dir
    if test_dir_path is None:
        # Fallback to config TEST_DIR
        test_dir_path = config.get('TEST_DIR', 'dataset/test_dataset')
    
    test_dir = pathlib.Path(test_dir_path)
    if not test_dir.exists():
        print(f"Error: Test directory '{test_dir}' does not exist.")
        sys.exit(1)
        
    print(f"Scanning test images in: {test_dir}")
    test_files = sorted(
        list(test_dir.rglob('*.JPG')) + list(test_dir.rglob('*.jpg'))
    )
    if not test_files:
        print("No test images (.jpg, .JPG) found in the test directory.")
        sys.exit(1)
    print(f"Found {len(test_files)} test images.")

    # Image normalization params
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    
    test_transform = A.Compose([
        A.Resize(model_input, model_input),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])

    test_ds = GCPTestDataset(test_files, test_dir, test_transform)
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0 if platform.system() == "Windows" else 4
    )

    predictions = {}
    print("Running TTA Inference...")
    for imgs, rel_paths, orig_ws, orig_hs in tqdm(test_loader):
        imgs = imgs.to(device)
        kp_norm, cls_probs = tta_predict(model, imgs)
        
        kp_norm = kp_norm.cpu()
        cls_idx = cls_probs.argmax(dim=1).cpu().tolist()

        for i, rp in enumerate(rel_paths):
            x_pred = float(kp_norm[i, 0] * orig_ws[i])
            y_pred = float(kp_norm[i, 1] * orig_hs[i])
            shape_name = idx2class.get(cls_idx[i], 'Cross')
            
            predictions[rp] = {
                'mark': {
                    'x': round(x_pred, 2),
                    'y': round(y_pred, 2)
                },
                'verified_shape': shape_name
            }

    print(f"Saving predictions to: {args.output}")
    # Create output directories if necessary
    out_path = pathlib.Path(args.output)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)

    print("Inference completed successfully!")

if __name__ == '__main__':
    main()
