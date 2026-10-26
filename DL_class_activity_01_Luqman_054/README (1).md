# CS4152 – Deep Learning and Neural Networks
## Class Activity 1: Building and Training Neural Networks with Keras and TensorFlow

| | |
|---|---|
| **Student** | Luqman Khan |
| **SID** | S202266054 |
| **Section** | V1 |
| **Course** | CS4152 – Deep Learning and Neural Networks, Fall 2026 |
| **Submitted to** | Dr. Jameel Ahmad |


---

## Overview

This project covers the practical side of neural networks with **Keras** and **TensorFlow**. It starts with a **regression** problem, moves to **binary** and **multiclass classification**, and finishes with **regularization** and **saving, loading and managing trained models**.

| Task | Name | Problem type | Dataset | Marks |
|---|---|---|---|---|
| 1 | Regression Model: Taxi Fare Prediction | Regression | NYC `taxi-fares.csv` | 30 |
| 2 | Binary Classification: Credit Card Fraud Detection | Binary classification | `creditcard.csv` | 30 |
| 3 | Multiclass Classification: Nonlinear Data Classification | Multiclass classification | Synthetic 2D data (4 classes) | 25 |
| 4 | Regularization Technique: Dropout Implementation | Regularization | Network from Task 3 | 10 |
| 5 | Model Persistence and Advanced Techniques | Saving, loading, callbacks | Model from Task 1 | 5 |

### Contents

1. [Project structure](#project-structure)
2. [Setup instructions](#setup-instructions)
3. [Task 1: Taxi Fare Prediction](#task-1-regression-model--taxi-fare-prediction)
4. [Task 2: Credit Card Fraud Detection](#task-2-binary-classification--credit-card-fraud-detection)
5. [Task 3: Multiclass Classification](#task-3-multiclass-classification--nonlinear-data-classification)
6. [Task 4: Dropout](#task-4-regularization-technique--dropout-implementation)
7. [Task 5: Model Persistence and Callbacks](#task-5-model-persistence-and-advanced-techniques)
8. [Overall summary](#overall-summary)

---

## Project structure

```
CS4152_Activity1_GroupX_LastNames/
├── README.md
├── DL__CA_01_Task_1.ipynb        # Task 1: Taxi fare regression
├── DL__CA_01_Task_2.ipynb        # Task 2: Credit card fraud detection
├── DL__CA_01_Task_3___4.ipynb    # Task 3 (multiclass) and Task 4 (dropout)
├── DL__CA_01_Task_5.ipynb        # Task 5: Saving, loading, callbacks
├── Data/
│   ├── taxi-fares.csv
│   └── creditcard.csv
├── results/                      # Created by the notebooks
│   ├── task1_history.csv
│   ├── task1_experiments.csv
│   └── task2_comparison.csv
└── saved_models/                 # Created by the Task 5 notebook
    ├── taxi_model.h5
    ├── taxi_model_savedmodel/
    ├── best_taxi_checkpoint.h5
    └── task5_history.csv
```

---

## Setup instructions

**1. Requirements**

- Python 3.10 or newer (the notebooks were run on Python 3.13)
- TensorFlow 2.21 (with the Keras 3 that comes with it)
- numpy, pandas, matplotlib, seaborn, scikit-learn

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

**2. Datasets**

The datasets are not included in the notebooks. Download them and place them where each notebook expects them.

| Dataset | Used in | Notes |
|---|---|---|
| `taxi-fares.csv` | Tasks 1 and 5 | NYC taxi fares, about 55,000 rows |
| `creditcard.csv` | Task 2 | 284,807 transactions, 31 columns |

The notebooks were written for Google Colab and read the files from `/content/` (for example `/content/taxi-fares.csv`). To run them locally, change those paths to your own folder, for example `Data/taxi-fares.csv`.

**3. How to run**

1. Open a notebook in Jupyter or Google Colab.
2. Run the cells **from top to bottom, in order**. Later cells depend on variables and functions created by earlier cells.
3. Task 5 trains the model itself, so it does not need Task 1 to be run first.

**4. Reproducibility**

All notebooks fix the random seed (`SEED = 42`, or `random_state` in scikit-learn) so results can be repeated. Neural networks still have some randomness during training, so your numbers may differ slightly from the ones in this README.

---

## Task 1: Regression Model – Taxi Fare Prediction

**Notebook:** `DL__CA_01_Task_1.ipynb` · **Marks:** 30

### Objective

Build and train a neural network that predicts the **fare of a New York City taxi trip** from simple trip information. Because the fare is a continuous number in dollars, this is a **regression** problem.

### Dataset and data preparation (part a)

The raw data has 55,000 trips with columns for the pickup time, pickup and dropoff coordinates, passenger count and fare. The notebook prepares it in these steps:

1. **Single-passenger trips only.** Keeping `passenger_count == 1` reduces the data from about 55,000 to **38,233 rows**.
2. **Drop unused columns** (`key` and `passenger_count`).
3. **Extract features from the pickup time:** `day_of_week` (0 = Monday to 6 = Sunday) and `pickup_time` (the hour, 0 to 23).
4. **Compute the distance** in miles from the coordinates. The longitude difference is multiplied by 54.6 and the latitude difference by 69.0 (approximate miles per degree in New York), then the straight-line distance is calculated: `distance = sqrt(dx² + dy²)`.
5. **Remove the raw date and coordinate columns.**
6. **Remove outliers:** keep distances between 1.0 and 10.0 miles and fares between $0 and $50. This leaves **23,298 rows**.

The final inputs are `x = [day_of_week, pickup_time, distance]` and the target is `y = fare_amount`.

### Network architecture (part b)

| Layer | Neurons | Activation | Parameters |
|---|---|---|---|
| Input | 3 | none | n/a |
| Hidden 1 | 512 | ReLU | 3 × 512 + 512 = 2,048 |
| Hidden 2 | 512 | ReLU | 512 × 512 + 512 = 262,656 |
| Output | 1 | none (regression) | 512 + 1 = 513 |
| **Total** | | | **265,217** |

The output layer has **no activation** because the fare can be any positive number. Two hidden layers of 512 neurons give the network enough capacity to learn the nonlinear link between distance, time and fare, while staying small enough to train quickly.

### Training (part c)

- Optimizer: **Adam**, loss: **MAE** (mean absolute error), metric: **MAE**
- 100 epochs, batch size 100, `validation_split = 0.2` (80% training, 20% validation)
- The training history is saved to `results/task1_history.csv`.

### Evaluation (part d)

| Result | Value |
|---|---|
| Final training MAE | 2.279 |
| Final validation MAE | 2.255 |
| R² score | 0.7512 |
| 2-mile trip, Friday 5 PM (day=4, hour=17) | **$10.66** |
| 2-mile trip, Saturday 5 PM (day=5, hour=17) | **$10.45** |
| Difference | $0.21 |

**What the results mean**

- **Learning curve.** Both MAE curves fall quickly in the first few epochs and then flatten at about $2.26–$2.28. The model has converged and trains stably.
- **No overfitting.** Training and validation MAE are almost equal. Validation MAE is slightly lower because Keras averages the training MAE over the whole epoch while the weights are still changing, whereas validation is measured at the end of the epoch.
- **Accuracy in dollars.** On average, a predicted fare is off by about **$2.26**. The model only knows day, hour and distance, so it cannot see traffic, the exact route or tolls.
- **R² = 0.75.** The model explains about 75% of the variation in fares. Note that this R² was calculated on the same data used for training, so it is not a perfect estimate for completely new trips.
- **Friday vs Saturday.** The Friday fare is slightly higher. This is plausible for New York: Friday at 5 PM is rush hour, and slower traffic increases the time-based part of the meter charge, while Saturday evening traffic is usually lighter. The gap is small because distance matters far more than the day of the week.

### Experiments and documentation (part e)

All experiments used **30 epochs** to keep the run time reasonable. The results are in `results/task1_experiments.csv`.

**Run-to-run variance (same settings, seeds 1, 2, 3)**

| Seed | Train MAE | Val MAE | R² | Time (s) |
|---|---|---|---|---|
| 1 | 2.317 | 2.324 | 0.735 | 64.9 |
| 2 | 2.313 | 2.305 | 0.743 | 65.5 |
| 3 | 2.319 | 2.297 | 0.740 | 63.0 |

Different random starts give slightly different results (validation MAE varies by about $0.03 and R² by about 0.008). This is normal for neural networks, so conclusions should come from averages and not from a single run.

**Hidden layer width (batch size 100, seed 1)**

| Width | Train MAE | Val MAE | R² | Time (s) |
|---|---|---|---|---|
| 128 | 2.315 | 2.313 | 0.736 | 25.8 |
| 256 | 2.311 | 2.318 | 0.736 | 35.9 |
| 512 | 2.317 | 2.324 | 0.735 | 66.9 |
| 1024 | 2.326 | 2.331 | 0.734 | 246.8 |

Making the layers wider did **not** improve accuracy. All widths perform about the same, but the 1024-neuron model takes about 10 times longer than the 128-neuron model. With only three input features, a small network is enough.

**Batch size (width 512, seed 1)**

| Batch size | Train MAE | Val MAE | R² | Time (s) |
|---|---|---|---|---|
| 32 | 2.320 | 2.313 | 0.743 | 136.3 |
| 100 | 2.317 | 2.324 | 0.735 | 61.0 |
| 500 | 2.314 | 2.320 | 0.733 | 36.4 |
| 1000 | 2.317 | 2.328 | 0.729 | 26.6 |

Smaller batches update the weights more often per epoch, so they take longer (136 s at batch size 32 versus 27 s at 1000) and give a slightly higher R². Larger batches are much faster but R² drops a little. A batch size of 100 is a good balance.

### Key takeaways

- A two-layer network reaches an average error of about $2.26 and R² of about 0.75.
- Extra width and very small batches cost a lot of time for almost no gain.
- Results change a little from run to run, so experiments should be repeated.

---

## Task 2: Binary Classification – Credit Card Fraud Detection

**Notebook:** `DL__CA_01_Task_2.ipynb` · **Marks:** 30

### Objective

Build a neural network that classifies each credit card transaction as **legitimate (0)** or **fraudulent (1)**. The main difficulty is that the data is **extremely imbalanced**, so the task is also about learning to evaluate a model correctly and handle imbalance.

### Data analysis and preparation (part a)

The dataset has **284,807 transactions** and 31 columns: `Time`, `V1` to `V28` (anonymized features), `Amount` and `Class`.

| Class | Count | Percentage |
|---|---|---|
| Legitimate (0) | 284,315 | 99.827% |
| Fraudulent (1) | 492 | **0.173%** |

A bar chart of the class distribution shows how rare fraud is. The preparation steps are:

1. Drop the `Time` column. This leaves **29 input features** (`V1`–`V28` and `Amount`).
2. Split into 80% training and 20% testing with `train_test_split(..., test_size=0.2, stratify=Y, random_state=0)`.

**Why stratified sampling matters.** With a random split, the 492 fraud cases could be spread unevenly, so the test set might get far fewer (or more) frauds than expected. Stratified sampling keeps the same class proportions in both sets. The notebook confirms this: fraud is **0.173%** in the training set and **0.172%** in the test set.

### Network architecture (part b)

| Layer | Neurons | Activation | Parameters |
|---|---|---|---|
| Input | 29 | none | n/a |
| Hidden | 128 | ReLU | 29 × 128 + 128 = 3,840 |
| Output | 1 | **Sigmoid** | 128 + 1 = 129 |
| **Total** | | | **3,969** |

- **Sigmoid** squeezes the output to a value between 0 and 1, which is read as the **probability of fraud**.
- **Binary cross-entropy** is the loss for two-class problems. It punishes confident wrong answers heavily.
- A single hidden layer of 128 neurons is enough for 29 numeric features.

### Training (part c)

Adam optimizer, binary cross-entropy loss, accuracy metric, **10 epochs**, batch size **100**. The test set is passed as `validation_data`. A prediction above 0.5 is treated as fraud.

### Evaluation: model without class weights (part d)

The notebook builds a confusion matrix and computes the main rates.

| | Predicted legitimate | Predicted fraud |
|---|---|---|
| **Actually legitimate** | TN = 56,856 | FP = 8 |
| **Actually fraud** | FN = 28 | TP = 70 |

| Metric | Value | Meaning |
|---|---|---|
| True Positive Rate (TPR) | **0.7143** | 71.4% of frauds were caught |
| False Positive Rate (FPR) | 0.000141 | Only 8 legitimate transactions were wrongly flagged |
| Accuracy | 0.99937 | Looks excellent, but see below |
| Precision (derived) | 0.897 | About 9 in 10 fraud alerts were real |

**Why accuracy alone is misleading.** About 99.83% of transactions are legitimate. A useless model that predicts "legitimate" for everything would score about 99.83% accuracy while catching **0%** of fraud. So high accuracy does not prove the model finds fraud. TPR, FPR and precision must be checked as well.

### Handling class imbalance with class weights (part e)

The model was retrained with `class_weight={0: 1.0, 1: 0.01}`, which tells the network that errors on fraud matter **100 times less** than errors on legitimate transactions.

| Model | TN | FP | FN | TP | TPR | FPR | Accuracy |
|---|---|---|---|---|---|---|---|
| No weights | 56,856 | 8 | 28 | 70 | 0.7143 | 0.00014 | 0.99937 |
| `{0: 1, 1: 0.01}` | 56,863 | 1 | 61 | 37 | 0.3776 | 0.00002 | 0.99891 |

**Trade-off.** The weights cut false alarms from 8 to 1, but missed frauds more than doubled (28 to 61), and the fraud detection rate fell from 71% to 38%. Note that this weight does the **opposite** of the usual strategy for improving fraud detection, which is why the result is worse for catching fraud.

### Analysis and business implications (part f)

- **Cost of errors.** A missed fraud (false negative) means direct money loss and chargeback costs. A false alarm (false positive) blocks a real customer, causing frustration and lost trust. A good system balances both costs.
- **Better weights in practice.** A real system would normally weight fraud **higher**, for example `{0: 1, 1: 10}`, and tune the value using the results.
- **Other ways to handle imbalance:** oversampling the minority class (for example SMOTE), undersampling the majority class, changing the decision threshold, using precision-recall curves, and using ensemble models such as Random Forest or XGBoost.

### Key takeaways

- With 0.173% fraud, accuracy is nearly useless on its own. Always look at the confusion matrix, TPR, FPR and precision.
- The unweighted model catches 71% of fraud with very few false alarms.
- Class weights are a tool for choosing a trade-off. The direction of the weight decides whether the model catches more fraud or raises fewer alarms.

---

## Task 3: Multiclass Classification – Nonlinear Data Classification

**Notebook:** `DL__CA_01_Task_3___4.ipynb` (Task 3 part) · **Marks:** 25

### Objective

Build a neural network that sorts 2D points into **four classes** and show how it makes decisions, using probabilities, a confusion matrix and a decision-boundary plot.

### Dataset preparation (part a)

The data is generated with scikit-learn's `make_blobs`:

- 1,000 points, 2 features (x and y coordinates), 4 clusters (classes 0 to 3)
- `cluster_std = 1.2`, `random_state = 42`
- Split 80/20 into **800 training** and **200 validation** points (stratified, so each class has 50 validation points)

A scatter plot colored by class shows the four clusters.

### Network architecture (part b)

| Layer | Neurons | Activation | Parameters |
|---|---|---|---|
| Input | 2 | none | n/a |
| Hidden | 128 | ReLU | 2 × 128 + 128 = 384 |
| Output | **4** | **Softmax** | 128 × 4 + 4 = 516 |
| **Total** | | | **900** |

**Why softmax is essential.** The output layer has one neuron per class. Softmax turns the four raw scores into four probabilities that are each between 0 and 1 and **add up to exactly 1**, using `softmax(z_i) = e^(z_i) / Σ e^(z_j)`. That is exactly what is needed when a point belongs to only one class. It also pairs naturally with the sparse categorical cross-entropy loss, which penalizes low probability on the true class.

### Training (part c)

Adam optimizer, `sparse_categorical_crossentropy` loss (labels are plain integers 0 to 3, so no one-hot encoding is needed), accuracy metric, **40 epochs**, batch size **10**, with the validation set passed as `validation_data`.

### Evaluation (part d)

1. **Accuracy curves.** Training and validation accuracy are plotted over the 40 epochs to check learning and overfitting.
2. **Predicting a point.** For the point **(0.2, 0.8)** the model outputs a probability for every class:

   | Class 0 | Class 1 | Class 2 | Class 3 |
   |---|---|---|---|
   | 0.146 | **0.790** | 0.042 | 0.021 |

   `np.argmax()` picks the largest probability, so the predicted class is **1**. The model is about 79% confident, and the other classes share the rest.
3. **Decision boundaries.** A grid of points covering the plane is passed through the model, and each grid point is colored by its predicted class. This draws the regions the network has learned, with the real data points on top.

### Analysis (part e)

The confusion matrix and classification report on the 200 validation points show **precision, recall and F1 of 1.00 for every class**, with accuracy of **1.00** (50 out of 50 correct in each class).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 | 50 |
| 1 | 1.00 | 1.00 | 1.00 | 50 |
| 2 | 1.00 | 1.00 | 1.00 | 50 |
| 3 | 1.00 | 1.00 | 1.00 | 50 |

**Which classes are harder?** Classes in the middle, or whose clusters overlap at the edges, are normally harder because points near a boundary get flatter probabilities (for example 0.48 versus 0.51). Here the clusters are well separated, so no class caused trouble.

**Ways to improve in harder cases** (from the written analysis):

- Add a second hidden layer (for example 128 → 64 neurons)
- Tune the width, or use regularization such as Dropout
- Standardize the inputs with `StandardScaler` or add polynomial features

### Key takeaways

- Softmax plus sparse categorical cross-entropy is the standard recipe for multiclass classification.
- One hidden layer of 128 neurons separates these four clusters perfectly.
- A perfect score on this dataset means the clusters are well separated. The same validation set was used while training and for the final check, so a separate test set would give a more honest estimate on harder data.

---

## Task 4: Regularization Technique – Dropout Implementation

**Notebook:** `DL__CA_01_Task_3___4.ipynb` (Task 4 part) · **Marks:** 10

> **Status note:** the uploaded notebook contains Task 3 only. The Task 4 dropout code, plots and results are not in the file yet, and Task 3's written analysis only mentions Dropout in passing. This section explains the task and what the finished notebook should contain. Remove this note once the Task 4 cells are added.

### Objective

Understand and apply **dropout**, a technique that reduces **overfitting** and helps a network generalize to new data.

### What dropout is

During training, a `Dropout(rate)` layer randomly switches off a fraction `rate` of the neurons' outputs in each training step. With `Dropout(0.2)`, about 20% of the signals are ignored each time. The network cannot depend on any single neuron, so it must learn more general patterns instead of memorizing the training data. At prediction time, dropout is turned off and the full network is used.

An everyday comparison is studying with some pages of a book randomly hidden each time. You learn the main ideas instead of memorizing every detail.

### Requirements

**(a) Implementation**

- Take the multiclass network from Task 3
- Add `Dropout(0.2)` after the hidden layer
- Retrain it and compare it with the model without dropout

```python
model = Sequential([
    Dense(128, activation='relu', input_dim=2),
    Dropout(0.2),
    Dense(4, activation='softmax')
])
```

**(b) Analysis**

- Plot training and validation accuracy for both models (with and without dropout)
- Compare the **gap** between training and validation accuracy. A large gap suggests overfitting.
- Decide whether dropout improved generalization
- Try dropout rates **0.1, 0.2 and 0.4** and record the findings

**(c) Documentation**

- Explain the dropout mechanism in your own words
- Describe when dropout helps (large networks, small datasets, a big training/validation gap)
- Describe the drawbacks of too much dropout (underfitting, slower learning, needing more epochs)

### What to expect

On the clean four-cluster data from Task 3, the network already reaches 100% validation accuracy, so dropout is unlikely to improve accuracy and the curves may be noisier. This is still a valid result to report: dropout is useful when a model overfits, and it is not always needed.

---

## Task 5: Model Persistence and Advanced Techniques

**Notebook:** `DL__CA_01_Task_5.ipynb` · **Marks:** 5

### Objective

Learn how to **save** a trained model, **load** it again, and manage training with **callbacks**. This is what makes a model reusable after the notebook is closed.

### Setup

The notebook rebuilds the Task 1 taxi-fare model: the same data preparation (23,298 trips), and the same architecture (3 inputs, two hidden layers of 512 ReLU neurons, 1 output). It fixes the seed and creates a `saved_models` folder.

### Callbacks (part c)

Two Keras callbacks are used while training for up to 100 epochs (batch size 100, validation split 0.2):

| Callback | Settings | What it does |
|---|---|---|
| `EarlyStopping` | `monitor='val_mae'`, `patience=8`, `restore_best_weights=True` | Stops training when validation MAE has not improved for 8 epochs and goes back to the best weights |
| `ModelCheckpoint` | `best_taxi_checkpoint.h5`, `monitor='val_mae'`, `save_best_only=True` | Saves the model to disk only when validation MAE improves |

**Results**

| Item | Value |
|---|---|
| Maximum epochs allowed | 100 |
| Epochs actually run | **86** |
| Epochs saved by early stopping | **14** |
| Best epoch | **78** |
| Best validation MAE | 2.2539 |
| Validation MAE in the last epoch | 2.2648 |

**How callbacks improve training efficiency**

1. **They save time.** Training stopped after 86 epochs because the model had stopped improving.
2. **They remove guesswork.** Validation error moves up and down, so the best number of epochs is hard to pick by hand.
3. **They return the best model, not the last one.** The best weights (MAE 2.2539) are better than those at the end of training (2.2648).
4. **They protect your work.** The checkpoint file keeps the best version even if training is interrupted or a later epoch is worse.

### Saving the model (part a)

```python
best_model.save('saved_models/taxi_model.h5')
best_model.export('saved_models/taxi_model_savedmodel')
```

| | H5 format | SavedModel format |
|---|---|---|
| Output | One `.h5` file | A folder (`saved_model.pb` plus `variables/`) |
| Created with | `model.save(...)` | `model.export(...)` in Keras 3 |
| Readable by | Python Keras apps only | Other frameworks and languages too (TensorFlow Serving, ML.NET and others) |
| Best for | Simple sharing and continued training | Deployment |

Both formats keep the same weights and biases, so the predictions are the same. The difference is portability. Keras 3 also warns that H5 is a legacy format and recommends the native `.keras` format.

### Loading and verification (part b)

The H5 model, the SavedModel and the checkpoint file are all loaded and used to predict four new trips. The H5 model is loaded with `compile=False` and then compiled again, which avoids a Keras 3 loading error with the `'mae'` loss name. The SavedModel is loaded with `tf.saved_model.load` and called through its `serve` function.

| Day | Hour | Distance (miles) | Original | Loaded H5 | Loaded SavedModel |
|---|---|---|---|---|---|
| 4 (Fri) | 17 | 2.0 | 10.5093 | 10.5093 | 10.5093 |
| 5 (Sat) | 17 | 2.0 | 10.3230 | 10.3230 | 10.3230 |
| 0 (Mon) | 9 | 5.0 | 24.0795 | 24.0795 | 24.0795 |
| 6 (Sun) | 23 | 8.0 | 29.7966 | 29.7966 | 29.7966 |

| Check | Result |
|---|---|
| Max difference, H5 vs original | 0.00000000 |
| Max difference, SavedModel vs original | 0.00000000 |
| Max difference, checkpoint vs original | 0.00000000 |
| `np.allclose` for all three | `True` |
| R² original model / loaded H5 model | 0.7491 / 0.7491 |

The loaded models give **exactly the same predictions** as the original. Loading restores the whole state of the network (weights and biases), so a saved model can be used without retraining, or trained further.

### Key takeaways

- Callbacks shorten training and keep the best model automatically.
- A model can be saved as H5 (simple, Python only) or SavedModel (portable, good for deployment).
- Saving and loading changes nothing. Predictions and R² are identical.

---

## Overall summary

| Task | Problem | Output layer | Loss | Main result |
|---|---|---|---|---|
| 1. Taxi Fare Prediction | Regression | 1 neuron, no activation | MAE | Validation MAE about $2.26, R² about 0.75 |
| 2. Credit Card Fraud Detection | Binary classification | 1 neuron, sigmoid | Binary cross-entropy | 71% of fraud caught, 8 false alarms; class weight 0.01 lowered fraud detection to 38% |
| 3. Multiclass Classification | 4-class classification | 4 neurons, softmax | Sparse categorical cross-entropy | 100% validation accuracy on well-separated clusters |
| 4. Dropout | Regularization | (Task 3 network plus Dropout) | Sparse categorical cross-entropy | Code and results still to be added to the notebook |
| 5. Model Persistence | Save, load, callbacks | (Task 1 network) | MAE | Early stopping at epoch 86, loaded models match exactly |

**Main lessons**

1. The output layer and loss function must match the problem: linear and MAE for regression, sigmoid and binary cross-entropy for two classes, softmax and sparse categorical cross-entropy for many classes.
2. A bigger network is not always better. In Task 1 wider layers cost much more time and gave no gain.
3. On imbalanced data, accuracy alone is misleading. Use the confusion matrix, TPR, FPR and precision.
4. Training is random, so results change slightly between runs. Fix the seed and repeat experiments.
5. Callbacks and model saving make training efficient and results reusable.
