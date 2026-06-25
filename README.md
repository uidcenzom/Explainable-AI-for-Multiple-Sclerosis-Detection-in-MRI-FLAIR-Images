# Explainable AI for Multiple Sclerosis Detection in MRI FLAIR Images

From CNN classification to post-hoc interpretability (Grad-CAM, Integrated Gradients, LIME).

**Authors:** Vincenzo Manzo, David Hartl
**Institution:** Universidad de Málaga (UMA) — *Aprendizaje Profundo* (Deep Learning), Semester 1
**Date:** January 2026

---

## Overview

Deep learning achieves strong performance in medical imaging, but clinical adoption is limited by the
black-box nature of neural networks: in healthcare, knowing *what* a model predicts is not enough —
we also need to understand *why*.

This project studies the applicability of Explainable AI (XAI) techniques to Multiple Sclerosis (MS)
identification from FLAIR MRI axial slices. A lightweight Convolutional Neural Network is trained to
classify slices as **MS** vs **control**, and three complementary post-hoc interpretability methods
are then applied and validated against ground-truth lesion masks.

The goal is deliberately **interpretability over pure accuracy**: the central question is not whether
the model classifies well, but whether its decisions are grounded in plausible lesion-related regions
rather than spurious artifacts.

---

## Key components

| | |
|---|---|
| **Dataset** | MSLesSeg — FLAIR sequence, 2D axial slices |
| **Framework** | PyTorch (with Captum for gradient-based attribution and LIME for perturbation-based explanations) |
| **Method** | Supervised learning followed by post-hoc interpretability analysis |
| **Environment** | Google Colab (CUDA GPU) with Google Drive for persistent storage |

---

## Dataset

The project uses the **MSLesSeg** dataset [1], restricted to the **axial FLAIR** view as required.

- Two groups: `MRIms_kde` (MS, label 1) and `MRIcontrol_kde` (controls, label 0).
- Official **patient-level** split into `imagesTr` (train) and `imagesTs` (test).
- Ground-truth lesion masks (`*_gt.png`) are available for MS slices.
- Masks are **never used as model input** — they are retained only for post-hoc XAI validation
  (IoU and pointing game).

### Data refinement

The data-loading strategy evolved across three iterations:

1. **`MSLesionDatasetCached`** — fast RAM caching; labels assigned purely from folder membership
   (MS vs control); returns `(image, label, mask)`.
2. **`MSLesionDatasetClean`** — intermediate refinement that inspects `_gt.png` masks to study
   slice-level label noise, keeping only MS slices that actually contain lesion pixels.
3. **`MSLesionDatasetFinal`** — consolidated dataset returning the `(image, label, mask)` triplet
   with **synchronized augmentation** (geometric transforms applied identically to image and mask).
   Control samples receive an all-zero mask to preserve a uniform interface.

### Preprocessing

- Resizing to a fixed input resolution (**128 × 218**, grayscale).
- **Skull stripping** via OpenCV intensity thresholding and bitwise masking, to suppress non-brain
  regions.
- Intensity normalisation.
- Training-time augmentation: random horizontal flips, small rotations (±15°), and mild
  brightness/contrast jitter (image only). Geometric transforms are applied synchronously to the mask
  to keep pixel-wise alignment.

---

## Model architecture (`MSClassifier`)

A deliberately simple CNN — simpler architectures are easier to debug and analyse with saliency
methods.

- **3 convolutional blocks:** `Conv2d → BatchNorm → ReLU → MaxPool`, with channel progression
  1 → 32 → 64 → 128.
- **Flatten:** 128 × 16 × 27 = 55,296 features.
- **2 fully connected layers:** 64 units (ReLU + Dropout 0.4) → 2 output logits.
- **Input:** single-channel grayscale MRI slice (128 × 218).
- **Output:** binary logits — class 0 (control) / class 1 (MS).

---

## Training setup

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| LR scheduler | `ReduceLROnPlateau` (×0.5 when validation loss stalls, patience = 2) |
| Loss | Weighted Cross-Entropy (class weights `[1.0, 1.5]`, higher for MS) |
| Batch size | 64 |
| Epochs | 20 |
| Reproducibility | Fixed seeds (PyTorch, NumPy, Python); deterministic CUDA |

A validation set (20%) is split from the training partition and used to drive learning-rate
scheduling and monitor generalisation. Overfitting is mitigated through augmentation, normalisation,
Batch Normalization, Dropout, and validation monitoring. The trained model is saved as a state dict
(`modello_MS_ottimizzato.pth`).

**Training behaviour.** Training is stable (≈15 s/epoch on GPU); the loss decreases from ≈0.11 to
≈0.069 and the scheduler reduces the learning rate from 10⁻³ to 2.5 × 10⁻⁴, enabling finer
convergence without clear instability.

---

## Classification performance

Although performance is not the primary objective, it provides context for the XAI analysis.

- **Accuracy:** 0.9982 on 3,853 test slices (≈ 7 misclassifications).
- **Support:** 2,366 healthy (class 0), 1,487 MS (class 1).
- Precision and recall are reported as 1.00 in the classification report (rounded to two decimals),
  reflecting an *almost perfect* — not strictly error-free — outcome.

**Important caveat.** Near-perfect classification does **not** automatically imply lesion-driven
reasoning. This must be examined with interpretability methods and mask-based agreement metrics.

---

## Interpretability methods

Three complementary XAI approaches are implemented:

1. **Grad-CAM** — coarse class-discriminative localisation using gradients flowing into the last
   convolutional layer (`conv3`); upsampled via bilinear interpolation.
2. **Integrated Gradients (IG)** — axiomatic pixel-level attribution (completeness), enhanced with
   **SmoothGrad** (noise tunnel) and **brain masking** to reduce noise and suppress background
   responses.
3. **LIME** — model-agnostic, perturbation-based local explanation using superpixels
   (quickshift / SLIC segmentation), as an independent sanity check.

---

## Quantitative XAI validation

Explanations are validated beyond visual inspection by comparing saliency maps with lesion masks:

- **IoU** — overlap between a thresholded explanation region and the ground-truth lesion mask.
- **Pointing game** — whether the peak attribution falls inside the lesion (strict) or within a
  dilated neighbourhood around it (relaxed).

**Results on lesion-positive slices (n = 1,487; LIME on a 20-slice subset):**

| Method | IoU | Relaxed hit | Strict hit |
|---|---:|---:|---:|
| Grad-CAM (brain-masked) | 0.0197 | 0.0868 | 0.0034 |
| Integrated Gradients (SmoothGrad + masking) | 0.0027 | 0.2495 | 0.2468 |
| LIME (20 lesions) | 0.0285 | 0.05 | 0.00 |

*(LIME also reports an intersection hit rate of 0.55.)*

**Takeaway.** Explanation–mask agreement is limited overall. SmoothGrad + brain masking substantially
improves IG peak alignment (higher pointing-game rates); LIME can show higher overlap on a small
subset but remains coarse and unstable. This is not necessarily contradictory: the task is *slice-level
classification*, whereas lesion masks are small focal targets, so attribution maps are expected to be
spatially diffuse and are therefore penalised by IoU. The defensible conclusion is that the classifier
distinguishes MS vs control effectively, but the evidence that it relies *primarily* on focal lesion
regions is mixed.


---

## Requirements

- Python 3 (Google Colab recommended)
- `torch`, `torchvision`
- `captum` — gradient-based attribution (Grad-CAM, Integrated Gradients)
- `lime`, `scikit-image` — perturbation-based explanations and superpixel segmentation
- `opencv-python` — image reading and preprocessing
- `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- `albumentations`, `segmentation-models-pytorch`, `tqdm`

```bash
pip install torch torchvision matplotlib opencv-python captum
pip install segmentation-models-pytorch
pip install lime
```


---

## Limitations and future work

- **Slice-level vs lesion-level.** The task is slice classification; XAI maps are not segmentation
  outputs, which limits IoU agreement with small focal masks.
- **Validation split.** The internal validation set is a *random slice-level* split of the training
  data (the official train/test split is patient-level, with explicit leakage checks). Slice-level
  validation may yield optimistic curves.
- **Future directions:**
  - patient-grouped validation splits for model selection;
  - causal perturbation tests (masking the lesion region vs a random region) to test whether lesion
    pixels are truly necessary for the prediction;
  - multi-task learning (classification + lesion segmentation) for stronger lesion-grounded reasoning.

---

## References

1. F. Guarnera et al., *MSLesSeg: Baseline and Benchmarking of a New Multiple Sclerosis Lesion
   Segmentation Dataset*, Scientific Data 12(1):920, 2025. doi:10.1038/s41597-025-05250-y
2. A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, 2019.
   arXiv:1912.01703
3. M. T. Ribeiro, S. Singh, C. Guestrin, *"Why Should I Trust You?": Explaining the Predictions of Any
   Classifier*, 2016. arXiv:1602.04938
4. R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based
   Localization*, ICCV 2017. doi:10.1109/ICCV.2017.74
5. M. Sundararajan, A. Taly, Q. Yan, *Axiomatic Attribution for Deep Networks*, 2017. arXiv:1703.01365
