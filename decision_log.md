# Aerial GCP Pose Estimation - Decision Log

This document log details the key engineering and architecture decisions made during the development of the GCP pose estimation pipeline.

---

## [DEC-01] Shift from Full-Image Regression to Crop-Based Pipeline
* **Context**: Initially, a full-image regression pipeline was implemented, where images (resolution $4096 \times 3068$ or $4096 \times 2730$) were resized to $384 \times 384$ or $640 \times 640$ pixels and fed into the network to directly regress `(x, y)` coordinates.
* **Problem**: Heavy downsampling caused the small Ground Control Point (GCP) markers (occupying less than $50 \times 50$ pixels originally) to completely lose structural clarity. The network failed to locate the markers and suffered from **center-bias collapse**, simply predicting coordinates near the average center of the dataset.
* **Mitigation**: Switched to a crop-based training pipeline. Localized $800 \times 800$ crops centered around the target were extracted during training and then resized. This preserved the high-resolution features and sub-pixel details of the markers.

---

## [DEC-02] Backbone Model Comparison and Selection
* **Context**: Needed an image feature extractor capable of parsing high-resolution local crops while maintaining low latency and memory footprints.
* **Options**:
  1. **EfficientNet-B3**: Standard CNN backbone with Compound Scaling.
  2. **ConvNeXt-Tiny**: Modernized CNN adopting Vision Transformer design principles (e.g., depthwise convolutions, LayerNorm, GELU, and inverted bottlenecks).
* **Decision**: While both models achieved near-perfect classification (Macro F1 $\approx 1.0$), **ConvNeXt-Tiny** exhibited faster convergence and slightly better localization stability under heavy augmentation. The final checkpoint saved in `checkpoints/best_model.pth` uses the ConvNeXt-Tiny backbone, but our inference script supports auto-detection of both architectures for maximum compatibility.

---

## [DEC-03] Adjusting Multi-Task Loss Weighting
* **Context**: The model solves a multi-task problem: shape classification and coordinate regression.
* **Initial Weighting**: $\mathcal{L}_{\text{total}} = 0.5 \times \mathcal{L}_{\text{localization}} + 1.5 \times \mathcal{L}_{\text{classification}}$
* **Problem**: Shape classification achieved $100\%$ accuracy within the first 5-10 epochs, but the localization metric (PCK) was extremely poor.
* **Decision**: Shifted loss weights to:
  $$\mathcal{L}_{\text{total}} = 2.0 \times \mathcal{L}_{\text{localization}} + 1.0 \times \mathcal{L}_{\text{classification}}$$
  This forced the network to dedicate most of its capacity to precise coordinate alignment.

---

## [DEC-04] Regression Loss Function Selection (Wing Loss)
* **Context**: Need a loss function for coordinate regression that is sensitive to small pixel offsets while remaining robust to label noise.
* **Decision**: Adopted **Wing Loss** over standard MSE/Smooth L1. Wing Loss uses a logarithmic function for small errors, which provides much stronger gradients when the prediction is close to the target center, resulting in sub-pixel alignment accuracy.

---

## [DEC-05] Robust Data Augmentations & Crop Jitter
* **Context**: Drone surveys capture imagery under a wide variety of lighting, angles, and surface conditions. The small dataset of 996 images is prone to overfitting.
* **Mitigation**:
  * **Jitter**: Applied a crop center jitter of up to $40\%$ (`JITTER_FRAC = 0.4`) during training, ensuring the marker is not always positioned exactly at the center of the crop.
  * **Augmentations**: Leveraged `albumentations` for random horizontal/vertical flips, 90-degree rotations, brightness/contrast adjustments, hue/saturation shifts, Gaussian noise, and Gaussian blur. This simulated variable flight altitudes and lighting environments.
