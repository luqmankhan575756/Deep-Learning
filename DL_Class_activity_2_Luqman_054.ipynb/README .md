# Deep Learning Assignment: Tasks 1 to 6

CNNs, data augmentation, fruit classification and detection, transfer learning and fine-tuning.

| | |
|---|---|
| **Name** | Luqman Khan |
| **SID** | S2024266054 |
| **Date** | 9 November 2026 |
| **Framework** | TensorFlow / Keras 2.21 |
| **Environment** | Google Colab (GPU) |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Contents](#2-repository-contents)
3. [Datasets](#3-datasets)
4. [Task Summaries](#4-task-summaries)
5. [Comparison with the Published Paper](#5-comparison-with-the-published-paper)
6. [How to Run](#6-how-to-run)
7. [Known Limitations](#7-known-limitations)
8. [Future Work](#8-future-work)
9. [Reference](#9-reference)

---

## 1. Overview

This project contains six notebooks and a written report. All numbers below are taken from the saved notebook outputs. Where a notebook was only partly executed, or where two notebooks used different settings, this is stated explicitly rather than filled in.

Percentages are accuracies unless noted. "pp" means percentage points.

| Task | Topic | Dataset | Main result | Status |
|:---:|---|---|---|---|
| 1 | CNN from scratch | MNIST | Test accuracy **98.93%** | Complete |
| 2 | Dense network vs CNN; one-factor experiments | CIFAR-10 | Dense 47.20%; CNN 68.82%; best CNN (dropout 0.25) **74.01%** | Complete |
| 3 | Data augmentation | CIFAR-10 | 71.99% to **79.29%** with standard augmentation | Complete |
| 4 | Fruit classification (RGB / Gray / HSV) and detection | Fruits-360 (278 classes) | RGB 94.98%, Gray 93.55%, HSV **95.43%**; detector best F1 0.277 | Complete |
| 5 | VGG16 transfer learning | Fruits-360 (277 classes) | Test accuracy **94.91%** (top-5 99.67%) | Partly run: full-data VGG16 only |
| 6 | Checkpoints, callbacks, fine-tuning, detector | Fruits-360 (120 classes) | 95.48% to **96.94%** after fine-tuning | Mostly run: detector evaluation not run |

---

## 2. Repository Contents

```
.
├── Task 1 notebook      CNN from scratch on MNIST
├── Task 2 notebook      Dense vs CNN, one-factor experiments on CIFAR-10
├── Task 3 notebook      Data augmentation on CIFAR-10
├── Task 4 notebook      Fruits-360 classification (RGB/Gray/HSV) and sliding-window detector
├── Task 5 notebook      VGG16 transfer learning on Fruits-360
├── Task 6 notebook      Callbacks, checkpointing, fine-tuning, 121-class detector
├── Deep_Learning_Report_Tasks_1-6.pdf    Written report with tables and figures
└── README.md            This file
```

> Rename the notebook entries above to match your actual file names before submitting.

---

## 3. Datasets

| Dataset | Used in | Details |
|---|---|---|
| **MNIST** | Task 1 | 60,000 training and 10,000 test images, 28x28 grayscale, scaled to [0, 1] |
| **CIFAR-10** | Tasks 2, 3 | 50,000 training and 10,000 test images, 32x32 colour. Task 3 splits 40,000 / 10,000 (stratified) for training / validation |
| **Fruits-360 (100x100)** | Tasks 4, 5, 6 | Downloaded with `kagglehub` (about 8.5 GB). Task 4: 278 classes, 119,572 train / 29,893 val / 49,785 test. Task 5: 277 classes (one class dropped because of a "BlackBerry 4" / "Blackberry 4" spelling mismatch). Task 6: 120-class subset plus a Background class |

---

## 4. Task Summaries

### Task 1: CNN from Scratch on MNIST

**Model:** Conv(64, 3x3) - MaxPool - Conv(64, 3x3) - MaxPool - Flatten - Dense(128) - Dense(10, softmax), **243,786** parameters (hand count matches `model.summary()`).
**Training:** Adam, sparse categorical cross-entropy, 5 epochs, batch size 128, 20% validation split.

| Train acc. | Val acc. | Test acc. | Test loss |
|:---:|:---:|:---:|:---:|
| 99.38% | 98.92% | **98.93%** | 0.0338 |

- Validation accuracy peaks at 99.00% in epoch 4; validation loss starts rising in epoch 5, so about four epochs are enough.
- The notebook also shows misclassified test images, the 64 first-layer filters and the first-layer feature maps.

### Task 2: Dense Network vs CNN on CIFAR-10

**Models:** Dense baseline (855,050 parameters) vs a three-block CNN with 32 / 64 / 128 filters (356,810 parameters). All runs use 10 epochs and a 20% validation split. One factor was changed at a time: batch size, dropout, BatchNorm, optimiser.

| Configuration | Test acc. (%) |
|---|:---:|
| Dense baseline | 47.20 |
| CNN (batch 32, Adam, no dropout) | 68.82 |
| Batch size 16 / 128 / 512 | 69.12 / 71.15 / 68.09 |
| Dropout 0.1 / **0.25** / 0.5 | 73.00 / **74.01** / 66.47 |
| BatchNorm + dropout 0.25 | 73.47 |
| SGD (lr 0.001) / SGD + momentum / RMSprop | 40.48 / 64.92 / 71.53 |

- The CNN is 21.6 pp more accurate than the dense network with 58% fewer parameters.
- Dropout 0.25 gave the best result (+5.19 pp over the baseline CNN).
- Most confused classes: cat/dog, deer/horse, truck/automobile.
- Each configuration was trained once (seed 42), so differences of 1 to 3 pp may be run-to-run noise.

### Task 3: Data Augmentation on CIFAR-10

The Task 2 baseline CNN was trained for 20 epochs in three ways.

| Model | Train acc. | Val acc. | Test acc. | Train-val gap | Time / epoch |
|---|:---:|:---:|:---:|:---:|:---:|
| No augmentation | 96.43 | 72.02 | 71.99 | 24.41 pp | 8.75 s |
| Standard (flip + 10% shift) | 81.05 | 79.19 | **79.29** | 1.86 pp | 28.20 s |
| Standard + rotation 15° + zoom 0.1 | 78.18 | 77.49 | 76.79 | 0.69 pp | 29.11 s |

- Standard augmentation gave the largest single gain in the assignment (**+7.30 pp**).
- Rotation and zoom did not help within 20 epochs; that model was still improving at epoch 20.
- **Caveat:** the notebook could not find the Task 2 results and used the baseline CNN (no dropout).

### Task 4: Fruit Classification and Detection on Fruits-360

**Model:** four Conv(5x5) + MaxPool blocks (16, 32, 64, 128 filters), Dense(1024), Dense(256), softmax. **5,323,702** parameters. Adam (lr 0.001), 25 epochs, batch size 50.

| Input | Train acc. | Val acc. | Test acc. | Top-5 test |
|---|:---:|:---:|:---:|:---:|
| RGB | 99.27 | 99.78 | 94.98 | 98.72 |
| Grayscale | 99.47 | 99.46 | 93.55 | 97.52 |
| HSV | 99.41 | 99.65 | **95.43** | 98.52 |

**Detector:** a Background class was added (279 outputs) and the RGB model was fine-tuned for 10 epochs on fruit images plus 1,000 background patches. Detection uses a sliding 100x100 window, per-class NMS (IoU 0.3) and a probability threshold. It was tested on 50 synthetic scenes (199 fruits). A detection is correct if the class matches and IoU is at least 0.5.

| Stride | Threshold | Precision | Recall | F1 | Time / scene |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 0.9 | 0.108 | 0.920 | 0.193 | 0.702 s |
| **20** | **0.9** | **0.176** | **0.648** | **0.277** | **0.283 s** |
| 40 | 0.9 | 0.227 | 0.281 | 0.251 | 0.121 s |

- Validation accuracy (about 99.7%) is much higher than test accuracy (about 95%). Fruits-360 images are video frames, so a random validation split contains near-duplicates. **Use the test set as the reliable figure.**
- Detection has good recall but very low precision (windows with partial fruits get confident wrong labels). YOLO or Faster R-CNN would be better choices.

### Task 5: Transfer Learning with VGG16

**Model:** VGG16 with frozen ImageNet weights (14,714,688 parameters) + Flatten + Dense(1000, ReLU) + 277-way softmax. 19,600,965 parameters in total, **4,886,277** trainable. 10 epochs, Adam, batch size 50, horizontal flip and 10% shift augmentation.

| Model | Epochs | Test acc. | Top-5 test | Time / epoch |
|---|:---:|:---:|:---:|:---:|
| VGG16 transfer | 10 | 94.91 | 99.67 | 491.1 s |
| Scratch RGB CNN (Task 4) | 25 | 94.98 | 98.72 | 53.2 s |

- On the full dataset, transfer learning matched the scratch CNN but was about 9.2x slower per epoch.
- The 10%-data experiment (where transfer learning should help most) and the final comparison table were written but **not executed**.
- The two rows come from different notebooks (different class sets and validation splits), so this is an indicative comparison only.

### Task 6: Model Saving, Callbacks and Fine-Tuning

**Setup:** 120-class subset plus a Background class (50,702 train / 12,620 val / 21,084 test). The notebook did not find the Task 5 model on Google Drive and trained a new VGG16 model.

| Stage | Epochs | Train acc. | Val acc. | Test acc. | Test loss |
|---|:---:|:---:|:---:|:---:|:---:|
| 1. Frozen base + callbacks | 10 | 98.60 | 93.95 | 95.48 | 1.305 |
| 2. Fine-tuned (VGG16 block 5) | 5 | 99.56 | 93.32 (best 95.78) | **96.94** | 0.807 |
| 3. 121-class detector | 5 | 99.38 | 95.37 | not evaluated | not evaluated |

- **Callbacks used:** ModelCheckpoint (best validation accuracy), EarlyStopping (patience 3), BackupAndRestore, CSVLogger.
- Fine-tuning block 5 (lr 1e-5) improved test accuracy by **+1.46 pp** and cut test loss by 38%.
- The reloaded best model gave identical predictions to the original (maximum difference 0.0).
- The planned detection evaluation (VGG16 detector vs Task 4 detector on 50 scenes) was **not executed**.

---

## 5. Comparison with the Published Paper

Reference paper: Muresan and Oltean, *Fruit recognition from images using deep learning* (arXiv:1712.00580).

| Input | This work | Notebook reference | Difference | Paper, Table 3 | Difference |
|---|:---:|:---:|:---:|:---:|:---:|
| RGB | 94.98 | 98.66 | -3.68 pp | 93.47 | +1.51 pp |
| Grayscale | 93.55 | 95.25 | -1.70 pp | 94.24 | -0.69 pp |
| HSV | 95.43 | 96.09 | -0.66 pp | 97.01 | -1.58 pp |

- Both studies find **HSV best**. Our results are within 1.6 pp of the paper's Table 3.
- The notebook reference values could not be found in the paper's Tables 3 or 4 and should be confirmed against the assignment sheet.
- Likely reasons for differences (not tested): our dataset has about 3x more classes (278 vs 90); the paper trained about four times longer with a decaying learning rate and dropout; random splits and initialisation differ.

---

## 6. How to Run

1. Open each notebook in **Google Colab** and select a GPU runtime (`Runtime > Change runtime type`).
2. Mount Google Drive when asked. Tasks 4 to 6 save a checkpoint to Drive after every epoch and can resume after a disconnect.
3. Install the requirements if they are missing:

   ```bash
   pip install tensorflow==2.21 numpy matplotlib scikit-learn kagglehub
   ```

4. Run the notebooks **in order, from Task 1 to Task 6**. Task 3 expects the Task 2 results and Task 6 expects the Task 5 model in memory or on Drive (see limitations below).
5. For Tasks 4 to 6, `kagglehub` downloads Fruits-360 (about 8.5 GB). Make sure Kaggle access is set up and there is enough disk space.

**Approximate time per epoch (Colab):** Task 4 about 53 s (RGB), Task 5 about 491 s, Task 6 about 255 to 270 s.

---

## 7. Known Limitations

- **Single runs.** Every configuration was trained once with seed 42; small differences (1 to 3 pp) may be noise.
- **Inconsistent class sets.** Task 4 uses 278 classes, Task 5 uses 277 and Task 6 uses 120, so results across Tasks 4 to 6 are not strictly comparable. The notebooks call the detector "121-class" even though in Task 4 it has 279 outputs.
- **Notebook dependencies.** Task 3 fell back to the baseline CNN because the Task 2 results were not in memory; Task 6 retrained VGG16 because the Task 5 model was not found.
- **Unreliable validation in Task 4.** Random validation splits on video frames leak near-duplicates; the test set is the trustworthy number.
- **Mixed timings.** HSV training in Task 4 was resumed across Colab sessions (111 s/epoch vs 53 s for RGB), so its timing is not comparable.
- **Incomplete runs.** Task 5: scratch-CNN, 10%-data and final-table sections not executed. Task 6: detector comparison and final scene visualisation not executed. These are reported as missing, not estimated.
- **Detection.** Synthetic scenes (fruits pasted on white) and low precision (below 23% in every setting).

---

## 8. Future Work

- Run the missing experiments: 10%-data comparison (Task 5) and the VGG16 vs Task 4 detector evaluation (Task 6).
- Use one class set and split by video sequence for Tasks 4 to 6.
- Repeat key experiments with several seeds and report mean and standard deviation.
- Re-run Task 3 with the best Task 2 model (dropout 0.25, BatchNorm) and train longer with rotation and zoom; try mixup / cutout and deeper networks such as ResNet.
- Reproduce the paper's best input (HSV + grayscale with hue/saturation changes and flips), and add dropout and a decaying learning rate to Task 4.
- Improve detection with class-agnostic NMS, hard-negative background examples and real annotated photographs; compare against YOLO or Faster R-CNN.

---

## 9. Reference

H. Muresan and M. Oltean, *Fruit recognition from images using deep learning*, Acta Univ. Sapientiae, Informatica 10(1), 26-42, 2018. arXiv:1712.00580.
