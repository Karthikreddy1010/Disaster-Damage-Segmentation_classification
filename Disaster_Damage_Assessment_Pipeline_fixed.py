#!/usr/bin/env python
# coding: utf-8

# # Disaster Damage Assessment Pipeline
# ## Stage 1: Building Segmentation + Stage 2: Damage Classification
# 
# This notebook implements an end-to-end deep learning pipeline for satellite image analysis:
# - **Stage 1**: U-Net++ with EfficientNet-B3 backbone for building segmentation
# - **Stage 2**: Dual-input ResNet50 for damage classification
# - **Inference**: Predict damage maps on new image pairs
# 
# Dataset: xView2 (pre/post disaster satellite imagery)

# ## Cell 1: Imports and Setup

# In[34]:


import os
import json
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from glob import glob
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchvision import models

# Check GPU
print(f'GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')


# ## Cell 2: Configuration

# In[35]:


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ── Data paths (EDIT THIS TO MATCH YOUR DATA LOCATION) ──────────
BASE_DIR = Path.home() / 'Satellite-based disaster damage'

print('BASE_DIR           :', BASE_DIR)
print('train/images exists:', (BASE_DIR / 'train' / 'images').exists())
print('train/labels exists:', (BASE_DIR / 'train' / 'labels').exists())
print('test/images  exists:', (BASE_DIR / 'test'  / 'images').exists())

train_img_dir = BASE_DIR / 'train' / 'images'
test_img_dir  = BASE_DIR / 'test'  / 'images'
train_files   = list(train_img_dir.glob('*')) if train_img_dir.exists() else []
test_files    = list(test_img_dir.glob('*'))  if test_img_dir.exists()  else []

print(f'\nTrain folder — {len(train_files)} total files')
print('  First 6:', [f.name for f in sorted(train_files)[:6]])
print(f'\nTest  folder — {len(test_files)} total files')
print('  First 6:', [f.name for f in sorted(test_files)[:6]])

assert BASE_DIR.exists(), f'DATA ROOT not found: {BASE_DIR}'
print('\n✓ All paths look good')


# In[36]:


class Config:
    # ── Paths ──────────────────────────────────────────────────────
    DATA_ROOT      = BASE_DIR
    TRAIN_IMG_DIR  = str(DATA_ROOT / 'train' / 'images')
    TRAIN_LBL_DIR  = str(DATA_ROOT / 'train' / 'labels')
    TEST_IMG_DIR   = str(DATA_ROOT / 'test'  / 'images')
    OUTPUT_DIR     = './outputs'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_DIR        = os.path.join(OUTPUT_DIR, 'logs')
    VIZ_DIR        = os.path.join(OUTPUT_DIR, 'visualizations')
    PRED_DIR       = os.path.join(OUTPUT_DIR, 'predictions')
    PATCH_DIR      = os.path.join(OUTPUT_DIR, 'patches')

    # ── Segmentation ───────────────────────────────────────────────
    SEG_IMG_SIZE   = 512
    SEG_BATCH_SIZE = 4
    SEG_LR         = 1e-4
    SEG_EPOCHS     = 100
    SEG_THRESHOLD  = 0.5
    SEG_MIN_AREA   = 100      # pixels — blobs smaller than this are removed

    # ── Classification ─────────────────────────────────────────────
    CLS_PATCH_SIZE  = 224
    CLS_BATCH_SIZE  = 16
    CLS_LR          = 1e-4
    CLS_EPOCHS      = 50
    CLS_NUM_CLASSES = 4

    # ── Damage label map (xView2 convention) ───────────────────────
    DAMAGE_CLASSES = {
        0: 'no-damage',
        1: 'minor-damage',
        2: 'major-damage',
        3: 'destroyed',
    }

    # Color overlays for visualization (BGR)
    DAMAGE_COLORS = {
        0: (0,   200,  0),    # green  – no damage
        1: (0,   200, 255),   # yellow – minor
        2: (0,   100, 255),   # orange – major
        3: (0,     0, 255),   # red    – destroyed
    }

    # ── Reproducibility ────────────────────────────────────────────
    SEED = 42

    # ── Device ─────────────────────────────────────────────────────
    DEVICE = 'cuda'   # falls back to cpu in train scripts if needed


def make_dirs():
    """Create all output directories."""
    for d in [Config.OUTPUT_DIR, Config.CHECKPOINT_DIR,
              Config.LOG_DIR, Config.VIZ_DIR, Config.PRED_DIR, Config.PATCH_DIR]:
        os.makedirs(d, exist_ok=True)
    print('✓ Output directories created')

make_dirs()


# ## Cell 3: Model Definitions

# In[37]:


# ─────────────────────────────────────────────────────────────────
# Stage 1 — U-Net++ with EfficientNet-B3 backbone
# ─────────────────────────────────────────────────────────────────

def build_segmentation_model(encoder_name: str = 'efficientnet-b3',
                              encoder_weights: str = 'imagenet',
                              in_channels: int = 3,
                              num_classes: int = 1) -> nn.Module:
    """
    U-Net++ segmentation model.
    Returns logits of shape (B, 1, H, W).
    Apply sigmoid + threshold at inference time.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,         # raw logits → use BCEWithLogitsLoss
        decoder_attention_type='scse',  # squeeze-and-excitation attention
    )
    return model

print('✓ Segmentation model builder defined')


# In[38]:


# ─────────────────────────────────────────────────────────────────
# Stage 2 — Dual-input Siamese ResNet50
# ─────────────────────────────────────────────────────────────────

class DualResNet50(nn.Module):
    """
    Siamese ResNet50 for damage classification.
    Both pre and post patches are passed through the SAME ResNet50 backbone
    (shared weights). Features are combined via absolute difference.
    """

    def __init__(self, num_classes: int = 4,
                 pretrained: bool = True,
                 dropout: float = 0.4):
        super().__init__()

        # ── Shared encoder (strip the classification head) ──────────
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )  # output: (B, 2048, 7, 7) for 224×224 input

        self.pool = nn.AdaptiveAvgPool2d(1)   # → (B, 2048, 1, 1)

        # ── Classification head ─────────────────────────────────────
        feat_dim = 2048
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled feature vector from a single image."""
        f = self.encoder(x)          # (B, 2048, 7, 7)
        f = self.pool(f)             # (B, 2048, 1, 1)
        f = f.flatten(1)             # (B, 2048)
        return f

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        f_pre  = self.encode(pre)
        f_post = self.encode(post)
        # Absolute difference → forces the network to learn *change*
        diff   = torch.abs(f_post - f_pre)   # (B, 2048)
        return self.classifier(diff)          # (B, num_classes)

print('✓ DualResNet50 model defined')


# In[39]:


# ─────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for binary segmentation."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(logits, targets)

        probs     = torch.sigmoid(logits)
        smooth    = 1e-6
        inter     = (probs * targets).sum(dim=(2, 3))
        dice_loss = 1 - (2 * inter + smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

print('✓ Loss functions defined')


# In[40]:


# ─────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────

device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

print('\n=== Segmentation model ===')
seg = build_segmentation_model()
x   = torch.randn(2, 3, 512, 512)
out = seg(x)
print(f'  Input : {x.shape}')
print(f'  Output: {out.shape}')   # expect (2, 1, 512, 512)
del seg, x, out

print('\n=== Classification model ===')
cls  = DualResNet50(num_classes=4)
pre  = torch.randn(4, 3, 224, 224)
post = torch.randn(4, 3, 224, 224)
out  = cls(pre, post)
print(f'  Input : pre {pre.shape}, post {post.shape}')
print(f'  Output: {out.shape}')   # expect (4, 4)
del cls, pre, post, out

print('\n✓ Both models initialised successfully')


# ## Cell 4: Dataset Utilities

# In[41]:


# ─────────────────────────────────────────────────────────────────
# Helper: build binary building mask from xView2 GeoJSON label
# ─────────────────────────────────────────────────────────────────

def json_to_mask(json_path: str, height: int = 1024, width: int = 1024) -> np.ndarray:
    """
    Parse an xView2 GeoJSON annotation file and rasterise all building
    polygons into a binary uint8 mask (0 = background, 255 = building).
    """
    with open(json_path) as f:
        data = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)

    for feat in data.get('features', {}).get('xy', []):
        geom = feat.get('wkt', '')
        if not geom.startswith('POLYGON'):
            continue
        coords_str = geom.replace('POLYGON ((', '').replace('))', '').strip()
        pts = []
        for pair in coords_str.split(','):
            pair = pair.strip()
            if pair:
                x, y = pair.split()
                pts.append([float(x), float(y)])
        if len(pts) < 3:
            continue
        poly = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)

    return mask


def json_to_damage_mask(json_path: str, height: int = 1024,
                         width: int = 1024) -> np.ndarray:
    """
    Parse a POST-disaster xView2 JSON and rasterise damage subtypes:
        no-damage   → 1
        minor-damage→ 2
        major-damage→ 3
        destroyed   → 4
        (0 = background / unknown)
    """
    subtype_map = {
        'no-damage':    1,
        'minor-damage': 2,
        'major-damage': 3,
        'destroyed':    4,
        'un-classified': 0,
    }

    with open(json_path) as f:
        data = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)

    for feat in data.get('features', {}).get('xy', []):
        geom     = feat.get('wkt', '')
        subtype  = feat.get('properties', {}).get('subtype', 'un-classified')
        label    = subtype_map.get(subtype, 0)

        if not geom.startswith('POLYGON') or label == 0:
            continue

        coords_str = geom.replace('POLYGON ((', '').replace('))', '').strip()
        pts = []
        for pair in coords_str.split(','):
            pair = pair.strip()
            if pair:
                x, y = pair.split()
                pts.append([float(x), float(y)])
        if len(pts) < 3:
            continue
        poly = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly], label)

    return mask

print('✓ Mask conversion functions defined')


# In[42]:


# ─────────────────────────────────────────────────────────────────
# Stage 1 — Segmentation Dataset
# ─────────────────────────────────────────────────────────────────

def get_seg_transforms(train: bool = True):
    if train:
        return A.Compose([
            A.Resize(Config.SEG_IMG_SIZE, Config.SEG_IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std =(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(Config.SEG_IMG_SIZE, Config.SEG_IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std =(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class SegmentationDataset(Dataset):
    """Returns (pre_image_tensor, binary_mask_tensor) pairs."""

    def __init__(self, img_dir: str, lbl_dir: str,
                 transform=None, img_size: int = 1024):
        self.img_dir   = Path(img_dir)
        self.lbl_dir   = Path(lbl_dir)
        self.transform = transform
        self.img_size  = img_size

        self.samples = []
        for img_path in sorted(self.img_dir.glob('*_pre_disaster.png')):
            stem      = img_path.stem
            json_path = self.lbl_dir / f'{stem}.json'
            if json_path.exists():
                self.samples.append((str(img_path), str(json_path)))

        print(f'SegmentationDataset: {len(self.samples)} samples found')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w  = image.shape[:2]

        mask  = json_to_mask(json_path, height=h, width=w)
        mask  = (mask > 0).astype(np.float32)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask'].unsqueeze(0)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.
            mask  = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

print('✓ Segmentation dataset defined')


# In[43]:


# ─────────────────────────────────────────────────────────────────
# Stage 2 — Classification Dataset (pre/post patch pairs)
# ─────────────────────────────────────────────────────────────────

def get_cls_transforms(train: bool = True):
    if train:
        return A.Compose([
            A.Resize(Config.CLS_PATCH_SIZE, Config.CLS_PATCH_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std =(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(Config.CLS_PATCH_SIZE, Config.CLS_PATCH_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std =(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class PatchClassificationDataset(Dataset):
    """Expects patch_dir/pre/, patch_dir/post/, patch_dir/labels.json"""

    def __init__(self, patch_dir: str, transform=None):
        self.patch_dir  = Path(patch_dir)
        self.transform  = transform
        self.pre_dir    = self.patch_dir / 'pre'
        self.post_dir   = self.patch_dir / 'post'
        labels_file     = self.patch_dir / 'labels.json'

        # ── Guard: raise a clear error if patch extraction was skipped ──
        if not labels_file.exists():
            raise FileNotFoundError(
                f"Patch labels not found at '{labels_file}'.\n"
                "You must run extract_all_patches() before train_classification().\n"
                "Example:\n"
                "  extract_all_patches(\n"
                "      img_dir    = Config.TRAIN_IMG_DIR,\n"
                "      lbl_dir    = Config.TRAIN_LBL_DIR,\n"
                "      output_dir = Config.PATCH_DIR,\n"
                "      use_gt_mask= True,\n"
                "  )"
            )

        with open(labels_file) as f:
            self.labels = json.load(f)

        self.ids = sorted(self.labels.keys())
        print(f'PatchClassificationDataset: {len(self.ids)} patch pairs')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        patch_id = self.ids[idx]
        label    = int(self.labels[patch_id])

        pre_path  = self.pre_dir  / f'{patch_id}.png'
        post_path = self.post_dir / f'{patch_id}.png'

        pre  = cv2.cvtColor(cv2.imread(str(pre_path)),  cv2.COLOR_BGR2RGB)
        post = cv2.cvtColor(cv2.imread(str(post_path)), cv2.COLOR_BGR2RGB)

        if self.transform:
            pre_aug  = self.transform(image=pre)
            post_aug = self.transform(image=post)
            pre  = pre_aug['image']
            post = post_aug['image']
        else:
            pre  = torch.from_numpy(pre.transpose(2,0,1)).float() / 255.
            post = torch.from_numpy(post.transpose(2,0,1)).float() / 255.

        return pre, post, torch.tensor(label, dtype=torch.long)

print('✓ Classification dataset defined')


# In[44]:


# ─────────────────────────────────────────────────────────────────
# Utility — compute class weights for imbalanced classification
# ─────────────────────────────────────────────────────────────────

def compute_class_weights(patch_dir: str,
                          num_classes: int = Config.CLS_NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    labels_file = Path(patch_dir) / 'labels.json'
    with open(labels_file) as f:
        labels = json.load(f)

    counts = np.zeros(num_classes, dtype=np.float32)
    for v in labels.values():
        counts[int(v)] += 1

    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (num_classes * counts)
    print('Class counts :', counts.astype(int))
    print('Class weights:', np.round(weights, 3))
    return torch.tensor(weights, dtype=torch.float32)

print('✓ Class weight computation defined')


# ## Cell 5: Patch Extraction (Bridge between Stage 1 and Stage 2)

# In[45]:


# ─────────────────────────────────────────────────────────────────
# Core extraction function (single image pair)
# ─────────────────────────────────────────────────────────────────

def extract_patches_from_pair(pre_img_path: str,
                               post_img_path: str,
                               mask: np.ndarray,
                               post_json_path: str = None,
                               patch_size: int = Config.CLS_PATCH_SIZE,
                               min_area: int = Config.SEG_MIN_AREA):
    """
    Given a binary mask and the pre/post image pair, find every connected
    building blob and crop the matching patches.
    """
    pre  = cv2.cvtColor(cv2.imread(pre_img_path),  cv2.COLOR_BGR2RGB)
    post = cv2.cvtColor(cv2.imread(post_img_path), cv2.COLOR_BGR2RGB)
    h, w = pre.shape[:2]

    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if post_json_path and Path(post_json_path).exists():
        damage_mask = json_to_damage_mask(post_json_path, height=h, width=w)
    else:
        damage_mask = np.full((h, w), fill_value=255, dtype=np.uint8)

    binary   = (mask > 0).astype(np.uint8) * 255
    num_comp, labels_cc, stats, _ = cv2.connectedComponentsWithStats(binary)

    patches = []

    for comp_id in range(1, num_comp):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[comp_id, cv2.CC_STAT_LEFT]
        y = stats[comp_id, cv2.CC_STAT_TOP]
        bw = stats[comp_id, cv2.CC_STAT_WIDTH]
        bh = stats[comp_id, cv2.CC_STAT_HEIGHT]

        margin = 10
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + bw + margin)
        y2 = min(h, y + bh + margin)

        pre_crop  = pre[y1:y2,  x1:x2]
        post_crop = post[y1:y2, x1:x2]

        if pre_crop.size == 0 or post_crop.size == 0:
            continue

        pre_patch  = cv2.resize(pre_crop,  (patch_size, patch_size))
        post_patch = cv2.resize(post_crop, (patch_size, patch_size))

        building_region = damage_mask[y1:y2, x1:x2]
        comp_mask       = (labels_cc[y1:y2, x1:x2] == comp_id)
        damage_vals     = building_region[comp_mask]
        valid           = damage_vals[(damage_vals >= 1) & (damage_vals <= 4)]

        if len(valid) > 0:
            damage = int(np.bincount(valid).argmax()) - 1
        else:
            damage = -1

        patches.append({
            'pre_patch':  pre_patch,
            'post_patch': post_patch,
            'damage':     damage,
            'bbox':       (x1, y1, x2 - x1, y2 - y1),
        })

    return patches

print('✓ Patch extraction defined')


# In[46]:


# ─────────────────────────────────────────────────────────────────
# Batch extractor — runs over entire dataset split
# ─────────────────────────────────────────────────────────────────

def extract_all_patches(img_dir: str,
                         lbl_dir: str,
                         output_dir: str,
                         mask_dir: str = None,
                         use_gt_mask: bool = True):
    """
    Extract building patches for every pre/post pair in img_dir.
    """
    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)
    out_dir = Path(output_dir)
    pre_out = out_dir / 'pre'
    post_out = out_dir / 'post'
    pre_out.mkdir(parents=True, exist_ok=True)
    post_out.mkdir(parents=True, exist_ok=True)

    labels_dict = {}
    patch_count = 0

    pre_images = sorted(img_dir.glob('*_pre_disaster.png'))
    print(f'Found {len(pre_images)} pre-disaster images to process...')

    for pre_path in tqdm(pre_images, desc='Extracting patches'):
        stem      = pre_path.stem
        post_name = stem.replace('_pre_', '_post_') + '.png'
        post_path = img_dir / post_name

        if not post_path.exists():
            continue

        if use_gt_mask:
            pre_json = lbl_dir / f'{stem}.json'
            if not pre_json.exists():
                continue
            img_tmp = cv2.imread(str(pre_path))
            h, w    = img_tmp.shape[:2]
            mask    = json_to_mask(str(pre_json), height=h, width=w)
        else:
            mask_path = Path(mask_dir) / f'{stem}_mask.png'
            if not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        post_json_name = stem.replace('_pre_', '_post_') + '.json'
        post_json      = lbl_dir / post_json_name

        patches = extract_patches_from_pair(
            pre_img_path=str(pre_path),
            post_img_path=str(post_path),
            mask=mask,
            post_json_path=str(post_json) if post_json.exists() else None,
        )

        for i, p in enumerate(patches):
            if p['damage'] == -1:
                continue

            patch_id = f'{stem}_{i:04d}'
            cv2.imwrite(str(pre_out  / f'{patch_id}.png'),
                        cv2.cvtColor(p['pre_patch'],  cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(post_out / f'{patch_id}.png'),
                        cv2.cvtColor(p['post_patch'], cv2.COLOR_RGB2BGR))
            labels_dict[patch_id] = p['damage']
            patch_count += 1

    labels_out = out_dir / 'labels.json'
    with open(labels_out, 'w') as f:
        json.dump(labels_dict, f)

    print(f'✓ Extracted {patch_count} labelled patches → {output_dir}')

    dist = Counter(labels_dict.values())
    for cls_id, name in Config.DAMAGE_CLASSES.items():
        print(f'  class {cls_id} ({name}): {dist.get(cls_id, 0)}')

print('✓ Batch patch extraction defined')


# ## Cell 6: Stage 1 — Segmentation Training

# In[47]:


# ─────────────────────────────────────────────────────────────────
# Segmentation Metrics
# ─────────────────────────────────────────────────────────────────

def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray,
              threshold: float = 0.5) -> float:
    pred = (pred_mask > threshold).astype(np.uint8)
    true = (true_mask  > threshold).astype(np.uint8)
    inter = (pred & true).sum()
    union = (pred | true).sum()
    return inter / (union + 1e-6)


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray,
               threshold: float = 0.5) -> float:
    pred = (pred_mask > threshold).astype(np.uint8)
    true = (true_mask  > threshold).astype(np.uint8)
    inter = (pred & true).sum()
    return (2 * inter) / (pred.sum() + true.sum() + 1e-6)

print('✓ Segmentation metrics defined')


# In[48]:


# ─────────────────────────────────────────────────────────────────
# Segmentation Training Helpers
# ─────────────────────────────────────────────────────────────────

def train_seg_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_seg(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_iou    = []
    all_dice   = []

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        with torch.cuda.amp.autocast():
            logits = model(images)
            loss   = criterion(logits, masks)

        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        gt    = masks.cpu().numpy()

        for p, g in zip(probs, gt):
            all_iou.append(iou_score(p[0], g[0]))
            all_dice.append(dice_score(p[0], g[0]))

    return (total_loss / len(loader),
            float(np.mean(all_iou)),
            float(np.mean(all_dice)))

print('✓ Segmentation training helpers defined')


# In[49]:


# ─────────────────────────────────────────────────────────────────
# Main Segmentation Training Loop
# ─────────────────────────────────────────────────────────────────

def train_segmentation():
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')

    # ── Dataset ──────────────────────────────────────────────────
    full_dataset = SegmentationDataset(
        img_dir   = Config.TRAIN_IMG_DIR,
        lbl_dir   = Config.TRAIN_LBL_DIR,
        transform = None,
    )

    n_total = len(full_dataset)
    n_val   = max(1, int(0.15 * n_total))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(Config.SEED)
    )

    train_ds.dataset.transform = get_seg_transforms(train=True)
    val_ds.dataset.transform   = get_seg_transforms(train=False)

    train_loader = DataLoader(train_ds, batch_size=Config.SEG_BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=Config.SEG_BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    # ── Model ────────────────────────────────────────────────────
    model     = build_segmentation_model().to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.SEG_LR,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SEG_EPOCHS, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()

    writer    = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, 'segmentation'))

    best_iou      = 0.0
    best_ckpt     = os.path.join(Config.CHECKPOINT_DIR, 'seg_best.pth')
    patience      = 7
    patience_ctr  = 0

    print(f'\n{"Epoch":>6} | {"Train Loss":>11} | {"Val Loss":>9} | '
          f'{"IoU":>6} | {"Dice":>6} | {"LR":>9}')
    print('-' * 65)

    for epoch in range(1, Config.SEG_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_seg_one_epoch(model, train_loader, optimizer,
                                         criterion, device, scaler)
        val_loss, iou, dice = validate_seg(model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t0
        print(f'{epoch:>6} | {train_loss:>11.4f} | {val_loss:>9.4f} | '
              f'{iou:>6.4f} | {dice:>6.4f} | {lr:>9.2e}  [{elapsed:.0f}s]')

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('Metric/IoU', iou,        epoch)
        writer.add_scalar('Metric/Dice',dice,       epoch)

        if iou > best_iou:
            best_iou     = iou
            patience_ctr = 0
            torch.save({
                'epoch':       epoch,
                'state_dict':  model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'best_iou':    best_iou,
            }, best_ckpt)
            print(f'  ✓ Best model saved  (IoU={best_iou:.4f})')
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break

    writer.close()
    print(f'\n✓ Segmentation training complete. Best IoU: {best_iou:.4f}')
    print(f'  Checkpoint: {best_ckpt}')
    return best_ckpt

print('✓ Segmentation training function defined')


# ### Run Segmentation Training
# 
# Uncomment and run the cell below to start Stage 1 training:

# In[50]:


# Uncomment to run Stage 1 training
seg_ckpt = train_segmentation()


# ## Cell 7: Stage 2 — Classification Training

# In[51]:


# ─────────────────────────────────────────────────────────────────
# Classification Metrics
# ─────────────────────────────────────────────────────────────────

def compute_cls_metrics(all_preds, all_labels, num_classes=Config.CLS_NUM_CLASSES):
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1     = f1_score(all_labels, all_preds, average='macro',
                      labels=list(range(num_classes)), zero_division=0)
    cm     = confusion_matrix(all_labels, all_preds,
                              labels=list(range(num_classes)))
    acc    = (all_preds == all_labels).mean()
    return f1, acc, cm

print('✓ Classification metrics defined')


# In[52]:


# ─────────────────────────────────────────────────────────────────
# Classification Training Helpers
# ─────────────────────────────────────────────────────────────────

def train_cls_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for pre, post, labels in loader:
        pre    = pre.to(device)
        post   = post.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(pre, post)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds       = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    f1, acc, _ = compute_cls_metrics(all_preds, all_labels)
    return total_loss / len(loader), f1, acc


@torch.no_grad()
def validate_cls(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for pre, post, labels in loader:
        pre    = pre.to(device)
        post   = post.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast():
            logits = model(pre, post)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds       = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    f1, acc, cm = compute_cls_metrics(all_preds, all_labels)
    return total_loss / len(loader), f1, acc, cm

print('✓ Classification training helpers defined')


# In[53]:


# ─────────────────────────────────────────────────────────────────
# Main Classification Training Loop
# ─────────────────────────────────────────────────────────────────

def train_classification(patch_dir: str = Config.PATCH_DIR):
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')

    # ── Dataset ──────────────────────────────────────────────────
    full_dataset = PatchClassificationDataset(
        patch_dir = patch_dir,
        transform = None,
    )

    n_total = len(full_dataset)
    n_val   = max(1, int(0.15 * n_total))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(Config.SEED)
    )

    train_ds.dataset.transform = get_cls_transforms(train=True)
    val_ds.dataset.transform   = get_cls_transforms(train=False)

    train_loader = DataLoader(train_ds, batch_size=Config.CLS_BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=Config.CLS_BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    # ── Class weights (imbalanced dataset) ───────────────────────
    class_weights = compute_class_weights(patch_dir).to(device)

    # ── Model ────────────────────────────────────────────────────
    model     = DualResNet50(num_classes=Config.CLS_NUM_CLASSES,
                             pretrained=True, dropout=0.4).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.CLS_LR,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.CLS_EPOCHS, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()

    writer    = SummaryWriter(
        log_dir=os.path.join(Config.LOG_DIR, 'classification'))

    best_f1      = 0.0
    best_ckpt    = os.path.join(Config.CHECKPOINT_DIR, 'cls_best.pth')
    patience     = 8
    patience_ctr = 0

    print(f'\n{"Epoch":>6} | {"T.Loss":>8} | {"T.F1":>6} | '
          f'{"V.Loss":>8} | {"V.F1":>6} | {"V.Acc":>6} | {"LR":>9}')
    print('-' * 72)

    for epoch in range(1, Config.CLS_EPOCHS + 1):
        t0 = time.time()

        t_loss, t_f1, t_acc = train_cls_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        v_loss, v_f1, v_acc, cm = validate_cls(
            model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t0
        print(f'{epoch:>6} | {t_loss:>8.4f} | {t_f1:>6.4f} | '
              f'{v_loss:>8.4f} | {v_f1:>6.4f} | {v_acc:>6.4f} | '
              f'{lr:>9.2e}  [{elapsed:.0f}s]')

        writer.add_scalar('Loss/train',    t_loss, epoch)
        writer.add_scalar('Loss/val',      v_loss, epoch)
        writer.add_scalar('F1/train',      t_f1,   epoch)
        writer.add_scalar('F1/val',        v_f1,   epoch)
        writer.add_scalar('Accuracy/val',  v_acc,  epoch)

        if v_f1 > best_f1:
            best_f1      = v_f1
            patience_ctr = 0
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'best_f1':    best_f1,
            }, best_ckpt)
            print(f'  ✓ Best model saved  (F1={best_f1:.4f})')
            print('  Confusion matrix:')
            for row in cm:
                print('   ', row.tolist())
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break

    writer.close()
    print(f'\n✓ Classification training complete. Best macro-F1: {best_f1:.4f}')
    print(f'  Checkpoint: {best_ckpt}')
    return best_ckpt

print('✓ Classification training function defined')


# ### Run Classification Training
# 
# Uncomment and run the cell below to start Stage 2 training (requires patches first):

# In[54]:


# ─────────────────────────────────────────────────────────────────
# Stage 1.5 + Stage 2  (run these together)
# Stage 1.5 must complete before Stage 2 or you will get a
# FileNotFoundError for outputs/patches/labels.json
# ─────────────────────────────────────────────────────────────────

# Step 1 — Extract building patches from training set
# Skip this block only if patches already exist in Config.PATCH_DIR
extract_all_patches(
    img_dir    = Config.TRAIN_IMG_DIR,
    lbl_dir    = Config.TRAIN_LBL_DIR,
    output_dir = Config.PATCH_DIR,
    use_gt_mask= True,   # set False to use seg-model predicted masks
)

# Step 2 — Train Stage 2 classifier on the extracted patches
cls_ckpt = train_classification()


# ## Cell 8: Inference Pipeline

# In[55]:


# ─────────────────────────────────────────────────────────────────
# Pre-processing helpers for inference
# ─────────────────────────────────────────────────────────────────

SEG_TRANSFORM = A.Compose([
    A.Resize(Config.SEG_IMG_SIZE, Config.SEG_IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

CLS_TRANSFORM = A.Compose([
    A.Resize(Config.CLS_PATCH_SIZE, Config.CLS_PATCH_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def preprocess_for_seg(image_rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, 512, 512) float tensor."""
    aug = SEG_TRANSFORM(image=image_rgb)
    return aug['image'].unsqueeze(0)


def preprocess_patch(patch_rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, 224, 224) float tensor."""
    aug = CLS_TRANSFORM(image=patch_rgb)
    return aug['image'].unsqueeze(0)

print('✓ Preprocessing helpers defined')


# In[56]:


# ─────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────

def load_seg_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = build_segmentation_model()
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    print(f'✓ Segmentation model loaded (IoU={ckpt.get("best_iou", "?"):.4f})')
    return model


def load_cls_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = DualResNet50(num_classes=Config.CLS_NUM_CLASSES, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    print(f'✓ Classification model loaded (F1={ckpt.get("best_f1", "?"):.4f})')
    return model

print('✓ Model loaders defined')


# In[57]:


# ─────────────────────────────────────────────────────────────────
# Stage 1 — predict building mask
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_mask(seg_model: torch.nn.Module,
                 image_rgb: np.ndarray,
                 device: torch.device,
                 threshold: float = Config.SEG_THRESHOLD) -> np.ndarray:
    """
    Returns binary mask (H, W) uint8 {0, 255} at original resolution.
    """
    orig_h, orig_w = image_rgb.shape[:2]
    inp    = preprocess_for_seg(image_rgb).to(device)

    with torch.cuda.amp.autocast():
        logit  = seg_model(inp)              # (1, 1, 512, 512)

    prob   = torch.sigmoid(logit).squeeze().cpu().numpy()  # (512, 512)
    mask   = (prob > threshold).astype(np.uint8) * 255

    mask   = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask

print('✓ Building mask prediction defined')


# In[58]:


# ─────────────────────────────────────────────────────────────────
# Stage 2 — classify each building patch
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def classify_patches(cls_model: torch.nn.Module,
                     patches: list,
                     device: torch.device) -> list:
    """
    Args:
        patches: list of dicts from extract_patches_from_pair().
    Returns:
        List of predicted damage class ints (0-3) in same order.
    """
    predictions = []

    for p in patches:
        pre_t  = preprocess_patch(p['pre_patch']).to(device)
        post_t = preprocess_patch(p['post_patch']).to(device)

        with torch.cuda.amp.autocast():
            logit = cls_model(pre_t, post_t)   # (1, 4)

        pred = logit.argmax(dim=1).item()
        predictions.append(pred)

    return predictions

print('✓ Patch classification defined')


# In[59]:


# ─────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────

def draw_predictions(post_image_rgb: np.ndarray,
                     patches: list,
                     predictions: list,
                     alpha: float = 0.5) -> np.ndarray:
    """
    Overlay coloured bounding boxes on the post-disaster image.
    Returns BGR image ready for cv2.imwrite.
    """
    vis = cv2.cvtColor(post_image_rgb, cv2.COLOR_RGB2BGR).copy()

    for p, pred in zip(patches, predictions):
        x, y, bw, bh = p['bbox']
        color = Config.DAMAGE_COLORS.get(pred, (128, 128, 128))
        label = Config.DAMAGE_CLASSES.get(pred, 'unknown')

        # Semi-transparent fill
        overlay = vis.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, -1)
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

        # Solid border
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

        # Label text
        cv2.putText(vis, label, (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    cv2.LINE_AA)

    return vis


def draw_legend(vis: np.ndarray) -> np.ndarray:
    """Add a damage class legend in the bottom-left corner."""
    legend_h = 30 * Config.CLS_NUM_CLASSES + 10
    legend_w = 180
    h, w     = vis.shape[:2]
    y0       = h - legend_h - 10
    x0       = 10

    # Semi-transparent background
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + legend_w, y0 + legend_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    for i, (cls_id, name) in enumerate(Config.DAMAGE_CLASSES.items()):
        color = Config.DAMAGE_COLORS[cls_id]
        cy    = y0 + 10 + i * 30
        cv2.rectangle(vis, (x0 + 5, cy), (x0 + 25, cy + 18), color, -1)
        cv2.putText(vis, name, (x0 + 32, cy + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis

print('✓ Visualization functions defined')


# In[60]:


# ─────────────────────────────────────────────────────────────────
# End-to-end single image pair inference
# ─────────────────────────────────────────────────────────────────

def run_inference(pre_path: str,
                  post_path: str,
                  seg_model: torch.nn.Module,
                  cls_model: torch.nn.Module,
                  device: torch.device,
                  output_dir: str = Config.PRED_DIR) -> dict:
    """
    Full pipeline for one image pair.

    Returns:
        {
            'mask':        np.ndarray (H, W) uint8,
            'predictions': list[int],
            'patches':     list[dict],
            'vis_path':    str,
        }
    """
    pre_rgb  = cv2.cvtColor(cv2.imread(pre_path),  cv2.COLOR_BGR2RGB)
    post_rgb = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)

    # Stage 1 — building mask
    print('  [1/3] Predicting building mask...')
    mask = predict_mask(seg_model, pre_rgb, device)

    # Stage 2a — extract patches using predicted mask
    print('  [2/3] Extracting building patches...')
    patches = extract_patches_from_pair(
        pre_img_path  = pre_path,
        post_img_path = post_path,
        mask          = mask,
        post_json_path= None,
    )
    print(f'         {len(patches)} buildings detected')

    if len(patches) == 0:
        print('  WARNING: No buildings found. Check segmentation quality.')
        return {'mask': mask, 'predictions': [], 'patches': [], 'vis_path': None}

    # Stage 2b — classify each building
    print('  [3/3] Classifying damage...')
    predictions = classify_patches(cls_model, patches, device)

    # Tally
    dist = Counter(predictions)
    for cls_id, name in Config.DAMAGE_CLASSES.items():
        print(f'         {name}: {dist.get(cls_id, 0)} buildings')

    # Visualization
    vis = draw_predictions(post_rgb, patches, predictions)
    vis = draw_legend(vis)

    stem     = Path(pre_path).stem.replace('_pre_disaster', '')
    vis_path = os.path.join(output_dir, f'{stem}_damage_map.png')
    cv2.imwrite(vis_path, vis)

    mask_path = os.path.join(output_dir, f'{stem}_building_mask.png')
    cv2.imwrite(mask_path, mask)

    print(f'  ✓ Visualization saved → {vis_path}')

    return {
        'mask':        mask,
        'predictions': predictions,
        'patches':     patches,
        'vis_path':    vis_path,
    }

print('✓ Single pair inference defined')


# In[61]:


# ─────────────────────────────────────────────────────────────────
# Batch inference over test set
# ─────────────────────────────────────────────────────────────────

def run_batch_inference(img_dir: str,
                        seg_ckpt: str,
                        cls_ckpt: str,
                        output_dir: str = Config.PRED_DIR):
    """Run inference on all pairs in img_dir."""
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')

    seg_model = load_seg_model(seg_ckpt, device)
    cls_model = load_cls_model(cls_ckpt, device)

    img_dir  = Path(img_dir)
    pre_imgs = sorted(img_dir.glob('*_pre_disaster.png'))
    print(f'\nRunning inference on {len(pre_imgs)} image pairs...\n')

    results = {}

    for pre_path in tqdm(pre_imgs, desc='Batch inference'):
        stem      = pre_path.stem
        post_name = stem.replace('_pre_', '_post_') + '.png'
        post_path = img_dir / post_name

        if not post_path.exists():
            continue

        print(f'Processing: {stem}')
        result = run_inference(str(pre_path), str(post_path),
                               seg_model, cls_model, device, output_dir)
        results[stem] = result['predictions']

    summary_path = os.path.join(output_dir, 'predictions_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f)
    print(f'\n✓ Batch inference complete. Summary → {summary_path}')

print('✓ Batch inference defined')


# ### Run Inference on a Single Image Pair
# 
# Uncomment and edit the paths below to run inference:

# In[64]:


# Example: Run inference on a single image pair
import os
from pathlib import Path

device   = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
seg_ckpt = './outputs/checkpoints/seg_best.pth'
cls_ckpt = './outputs/checkpoints/cls_best.pth'

seg_model = load_seg_model(seg_ckpt, device)
cls_model = load_cls_model(cls_ckpt, device)

# ── Pick a real pair from your training set ──────────────────────
# (test set may only have post images based on your Cell 2 output,
#  so use a training pair instead)
train_img_dir = Path(Config.TRAIN_IMG_DIR)

# Auto-pick the first available pre/post pair
pre_path  = str(sorted(train_img_dir.glob('*_pre_disaster.png'))[0])
post_path = pre_path.replace('_pre_disaster', '_post_disaster')

print(f'pre  → {pre_path}')
print(f'post → {post_path}')
assert os.path.exists(pre_path),  f'pre image not found:  {pre_path}'
assert os.path.exists(post_path), f'post image not found: {post_path}'

result = run_inference(
    pre_path=pre_path,
    post_path=post_path,
    seg_model=seg_model,
    cls_model=cls_model,
    device=device,
)


# ## Cell 9: Full Pipeline Runner

# In[65]:


# ─────────────────────────────────────────────────────────────────
# Full Pipeline Execution
# ─────────────────────────────────────────────────────────────────

def run_full_pipeline(use_gt_mask: bool = True):
    """
    Run the entire pipeline:
    1. Train segmentation (Stage 1)
    2. Extract patches
    3. Train classification (Stage 2)
    4. Run batch inference on test set
    """
    print('\n' + '═' * 60)
    print('  🚀 FULL DISASTER DAMAGE ASSESSMENT PIPELINE')
    print('═' * 60)

    # Stage 1
    print('\n' + '═' * 60)
    print('  STAGE 1 — Segmentation (U-Net++ / EfficientNet-B3)')
    print('═' * 60)
    seg_ckpt = train_segmentation()

    # Patch extraction
    print('\n' + '═' * 60)
    print('  STAGE 1.5 — Patch Extraction')
    print('═' * 60)
    extract_all_patches(
        img_dir    = Config.TRAIN_IMG_DIR,
        lbl_dir    = Config.TRAIN_LBL_DIR,
        output_dir = Config.PATCH_DIR,
        use_gt_mask= use_gt_mask,
    )

    # Stage 2
    print('\n' + '═' * 60)
    print('  STAGE 2 — Classification (Dual ResNet50)')
    print('═' * 60)
    cls_ckpt = train_classification(patch_dir=Config.PATCH_DIR)

    # Pipeline complete
    print('\n' + '═' * 60)
    print('  ✓ PIPELINE COMPLETE')
    print('═' * 60)
    print(f'  Segmentation checkpoint : {seg_ckpt}')
    print(f'  Classification checkpoint: {cls_ckpt}')
    print(f'  Outputs                 : {Config.OUTPUT_DIR}')

    # Batch inference if test dir exists
    if os.path.isdir(Config.TEST_IMG_DIR):
        print('\n  Running inference on test set...')
        run_batch_inference(Config.TEST_IMG_DIR, seg_ckpt, cls_ckpt)

    return seg_ckpt, cls_ckpt

print('✓ Full pipeline runner defined')


# ### Execute Full Pipeline
# 
# Uncomment and run the cell below to execute the complete pipeline:

# In[66]:


# ⚠️  MAIN EXECUTION — Uncomment to run full pipeline
seg_ckpt, cls_ckpt = run_full_pipeline(use_gt_mask=True)


# ## Cell 10: Summary and Next Steps

# In[67]:


print("""
╔═══════════════════════════════════════════════════════════════════╗
║     DISASTER DAMAGE ASSESSMENT PIPELINE — NOTEBOOK SUMMARY        ║
╚═══════════════════════════════════════════════════════════════════╝

✓ All components have been defined and are ready to use:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: SEGMENTATION (Building Detection)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Function:    train_segmentation()
  Model:       U-Net++ with EfficientNet-B3 backbone
  Input:       Pre-disaster satellite images (512×512)
  Output:      Binary building mask
  Loss:        Dice + BCE
  Metric:      IoU (Intersection over Union)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STAGE 1.5: PATCH EXTRACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Function:    extract_all_patches()
  Process:     Extract individual building patches from pre/post pairs
  Uses:        Building mask + damage labels from post-disaster JSON
  Output:      Organized patch directory with labels.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STAGE 2: CLASSIFICATION (Damage Assessment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Function:    train_classification()
  Model:       Siamese ResNet50 (dual-input)
  Input:       Pre + post building patches (224×224 each)
  Output:      Damage class (0-3):
               • 0: no-damage    (green)
               • 1: minor-damage (yellow)
               • 2: major-damage (orange)
               • 3: destroyed    (red)
  Loss:        CrossEntropyLoss with class weights
  Metric:      Macro-averaged F1 score

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Single pair:      run_inference(pre_path, post_path, seg_model, cls_model, device)
  Batch:            run_batch_inference(img_dir, seg_ckpt, cls_ckpt)
  Full pipeline:    run_full_pipeline(use_gt_mask=True)

  Output:           Damage maps with color-coded overlays + legend

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 QUICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Edit Cell 2 (Config) → Update BASE_DIR to point to your xView2 dataset

2. Run entire notebook sequentially (Cells 1–9 define components)

3. Choose your workflow:

   Option A: Run full pipeline
   ─────────────────────────────
   • Uncomment the last line in Cell 9
   • Executes all stages automatically

   Option B: Run individual stages
   ─────────────────────────────────
   • Cell 6: train_segmentation()
   • Cell 5: extract_all_patches()
   • Cell 7: train_classification()
   • Cell 8: run_batch_inference() or run_inference()

   Option C: Run inference only
   ──────────────────────────────
   • Load pre-trained checkpoints
   • Uncomment inference example in Cell 8
   • Provide pre/post image paths

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 KEY PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Segmentation
  • SEG_IMG_SIZE    = 512 (resize before processing)
  • SEG_EPOCHS      = 30
  • SEG_BATCH_SIZE  = 4
  • SEG_THRESHOLD   = 0.5 (mask probability cutoff)

  Classification
  • CLS_PATCH_SIZE  = 224
  • CLS_EPOCHS      = 25
  • CLS_BATCH_SIZE  = 16
  • CLS_NUM_CLASSES = 4

  Patch Extraction
  • SEG_MIN_AREA    = 100 (pixels, ignore smaller blobs)
  • CLS_PATCH_SIZE  = 224 (output patch size)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXPECTED DIRECTORY STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ~/Satellite-based disaster damage/
  ├── train/
  │   ├── images/
  │   │   ├── <disaster>_<id>_pre_disaster.png
  │   │   └── <disaster>_<id>_post_disaster.png
  │   └── labels/
  │       ├── <disaster>_<id>_pre_disaster.json
  │       └── <disaster>_<id>_post_disaster.json
  └── test/
      └── images/
          ├── <disaster>_<id>_pre_disaster.png
          └── <disaster>_<id>_post_disaster.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OUTPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  outputs/
  ├── checkpoints/
  │   ├── seg_best.pth
  │   └── cls_best.pth
  ├── logs/
  │   ├── segmentation/
  │   └── classification/
  ├── patches/
  │   ├── pre/     (pre-disaster patches)
  │   ├── post/    (post-disaster patches)
  │   └── labels.json
  ├── predictions/
  │   ├── <id>_damage_map.png
  │   ├── <id>_building_mask.png
  │   └── predictions_summary.json
  └── visualizations/

""")


# In[ ]:




