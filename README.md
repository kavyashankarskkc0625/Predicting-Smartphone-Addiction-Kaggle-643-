# Predicting Smartphone Addiction

Project skeleton for modeling and analyzing smartphone addiction risk.
# 📱 Phone Usage Addiction Prediction

A machine learning project for predicting whether a person is likely to be classified as **phone-use addicted** based on behavioral, screen-time, productivity, sleep, notification, app-usage, stress, gender, and academic-impact features.

The project uses feature engineering and CatBoost classification, with **ROC-AUC** as the primary evaluation metric.

---

## 📌 Project Overview

The goal is to build a binary classification model that predicts the probability of:

- `0` → Not addicted
- `1` → Addicted

The competition evaluates predictions using **Area Under the Receiver Operating Characteristic Curve (ROC-AUC)**.

Because the evaluation metric is ROC-AUC, the model outputs **probabilities** rather than hard 0/1 predictions.

---

## 📊 Dataset

The dataset contains separate training and test files.

### Original dimensions

| Dataset | Rows | Columns |
|---|---:|---:|
| Train | 691,369 | 14 |
| Test | 296,302 | 13 |
| Sample Submission | 296,302 | 2 |

### Training columns

```text
id
age
daily_screen_time_hours
social_media_hours
gaming_hours
work_study_hours
sleep_hours
notifications_per_day
app_opens_per_day
weekend_screen_time
gender
stress_level
academic_work_impact
addicted_label
```

### Test columns

```text
id
age
daily_screen_time_hours
social_media_hours
gaming_hours
work_study_hours
sleep_hours
notifications_per_day
app_opens_per_day
weekend_screen_time
gender
stress_level
academic_work_impact
```

The target column is:

```text
addicted_label
```

---

# 🔧 Feature Engineering

Nine additional behavioral features were engineered.

## 1. Total Entertainment Hours

Combines entertainment-oriented usage:

```text
total_entertainment_hours
```

This captures the overall amount of recreational phone activity.

---

## 2. Weekend Difference

Measures the difference between weekend and regular screen usage:

```text
weekend_difference
```

This attempts to capture changes in phone behavior during weekends.

Observed statistics:

```text
count    517906
mean          1.837633
std           1.766462
min          -7.91
max          11.49
```

---

## 3. Entertainment Ratio

Measures entertainment usage relative to overall usage.

```text
entertainment_ratio
```

Observed range:

```text
min ≈ 0.0039
max ≈ 0.9470
```

---

## 4. Social Media Ratio

Measures the proportion of usage associated with social media:

```text
social_media_ratio
```

Observed range:

```text
min = 0
max ≈ 0.8853
```

---

## 5. Gaming Ratio

Measures gaming usage relative to the relevant total:

```text
gaming_ratio
```

Observed range:

```text
min = 0
max ≈ 0.7995
```

---

## 6. Work/Study Ratio

Measures work/study usage relative to overall available time:

```text
work_study_ratio
```

Observed range:

```text
min = 0
max ≈ 0.9282
```

---

## 7. Leisure-to-Work Ratio

Measures leisure usage compared with work/study usage:

```text
leisure_to_work_ratio
```

Observed statistics:

```text
count    451821
mean          2.319269
std           2.018603
min           0.007421
max          21.083333
```

This feature contained substantial missingness and was therefore imputed during preprocessing.

---

## 8. Screen-Time Waking Ratio

Measures daily screen time relative to waking time:

```text
screen_time_waking_ratio
```

Observed statistics:

```text
count    559350
mean          0.446800
std           0.163338
min           0.025667
max           0.789141
```

Correlation with the target:

```text
≈ 0.6023
```

---

## 9. Non-Screen Time

Estimated time available outside sleep and daily screen time:

```text
non_screen_time = 24 - sleep_hours - daily_screen_time_hours
```

This provides an estimate of the amount of time not spent sleeping or using a screen.

---

# 🧪 Feature Engineering Summary

After feature engineering:

```text
Train Shape: (691369, 23)
Test Shape : (296302, 22)
```

The nine engineered features were:

```text
total_entertainment_hours
weekend_difference
entertainment_ratio
social_media_ratio
gaming_ratio
work_study_ratio
leisure_to_work_ratio
screen_time_waking_ratio
non_screen_time
```

---

# 🔍 Missing Values

The dataset contains substantial missing data.

Total missing values observed:

```text
Train: 2,004,513
Validation: 500,801
Test: 1,040,547
```

Missing values were handled through preprocessing rather than dropping rows.

### Numerical features

Median imputation was used:

```python
SimpleImputer(strategy="median")
```

### Categorical features

Most-frequent imputation was used:

```python
SimpleImputer(strategy="most_frequent")
```

---

# 🧩 Features Used by the Model

The final modeling dataset contained:

## Numerical Features

```text
age
daily_screen_time_hours
social_media_hours
gaming_hours
work_study_hours
sleep_hours
notifications_per_day
app_opens_per_day
weekend_screen_time
total_entertainment_hours
weekend_difference
entertainment_ratio
social_media_ratio
gaming_ratio
work_study_ratio
leisure_to_work_ratio
screen_time_waking_ratio
non_screen_time
```

Total numerical features:

```text
18
```

## Categorical Features

```text
gender
stress_level
academic_work_impact
```

Total categorical features:

```text
3
```

The `id` column was excluded from the final model features.

---

# ⚙️ Preprocessing

A `ColumnTransformer` was used.

### Numerical pipeline

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
```

### Categorical pipeline

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

This produced:

```text
Processed training features: 26
```

The processed shapes were:

```text
X_train processed: (553095, 26)
X_valid processed: (138274, 26)
X_test processed : (296302, 26)
```

For the CatBoost model, the categorical variables were handled directly and the final CatBoost feature matrix contained:

```text
X_train: (553095, 21)
X_valid: (138274, 21)
```

---

# ✂️ Train/Validation Split

The training data was split into:

```text
Training set   : 553,095 rows
Validation set : 138,274 rows
```

The target distribution was preserved.

### Original

```text
Class 1: 70.9424%
Class 0: 29.0576%
```

### Training

```text
Class 1: 70.9424%
Class 0: 29.0576%
```

### Validation

```text
Class 1: 70.9425%
Class 0: 29.0575%
```

This confirms that the split maintained the original class balance.

---

# 📈 Evaluation Metric

The competition uses:

## ROC-AUC

ROC-AUC measures how well the model ranks positive examples above negative examples.

The model therefore predicts:

```text
P(addicted_label = 1)
```

rather than using a fixed classification threshold.

This is important because ROC-AUC evaluates the ranking of prediction probabilities.

---

# 🤖 Baseline Model — Logistic Regression

A Logistic Regression model was trained as the initial baseline.

Validation ROC-AUC:

```text
0.917445
```

This provided a useful benchmark for evaluating more powerful nonlinear models.

---

# 🌳 CatBoost Model

CatBoost was then used as the main nonlinear model.

CatBoost was selected because it:

- Handles nonlinear relationships well
- Works effectively with mixed feature types
- Can model feature interactions
- Is strong on tabular datasets
- Provides feature importance
- Can handle categorical features directly

Installed version:

```text
CatBoost 1.2.10
```

---

# 🧪 Model Experiments

Several CatBoost depths were tested.

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.917445 |
| CatBoost depth=6 | 0.957701 |
| CatBoost depth=8 | 0.959760 |
| CatBoost depth=9 | 0.960385 |
| CatBoost depth=10, 1000 iterations | 0.960802 |
| CatBoost depth=10, 2000 iterations | **0.962327** |

The best validation score obtained during experimentation was:

```text
ROC-AUC = 0.962327
```

---

# 🏆 Best Model

The final selected model was:

```text
CatBoost
depth = 10
iterations = 2000
```

The validation result was:

```text
CatBoost depth=10 ROC-AUC: 0.962327
```

Improvement over the previous depth-9 result:

```text
+0.001943
```

Improvement over Logistic Regression:

```text
+0.044882
```

This was a substantial improvement over the baseline.

---

# 📉 CatBoost Training Progress

The 2000-iteration model showed continued improvement throughout training.

Selected validation results:

```text
Iteration 0     : 0.9149785
Iteration 100   : 0.9424866
Iteration 200   : 0.9484412
Iteration 300   : 0.9524510
Iteration 400   : 0.9552254
Iteration 500   : 0.9570006
Iteration 600   : 0.9582475
Iteration 700   : 0.9592303
Iteration 800   : 0.9597732
Iteration 900   : 0.9603069
Iteration 1000  : 0.9608057
Iteration 1100  : 0.9611168
Iteration 1200  : 0.9613503
Iteration 1300  : 0.9615841
Iteration 1400  : 0.9617664
Iteration 1500  : 0.9619170
Iteration 1600  : 0.9620414
Iteration 1700  : 0.9621298
Iteration 1800  : 0.9622433
Iteration 1900  : 0.9622861
Iteration 1999  : 0.9623274
```

The model was still improving at 2000 iterations, although the improvements became smaller.

---

# 🔎 CatBoost Feature Importance

The most important features from the final CatBoost experiment were:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | weekend_screen_time | 24.139255 |
| 2 | daily_screen_time_hours | 21.981835 |
| 3 | notifications_per_day | 12.128860 |
| 4 | app_opens_per_day | 10.912877 |
| 5 | social_media_hours | 10.338395 |
| 6 | social_media_ratio | 5.319683 |
| 7 | work_study_ratio | 3.411355 |
| 8 | gaming_ratio | 2.436313 |
| 9 | work_study_hours | 1.714277 |
| 10 | screen_time_waking_ratio | 1.254266 |
| 11 | non_screen_time | 1.251453 |
| 12 | weekend_difference | 0.961167 |
| 13 | entertainment_ratio | 0.944029 |
| 14 | age | 0.917947 |
| 15 | gaming_hours | 0.752419 |

The strongest predictors were primarily related to:

- Screen time
- Weekend phone usage
- Notifications
- Number of app openings
- Social media usage
- Usage ratios

This suggests that direct phone-use behavior carries substantially more predictive information than demographic variables in this dataset.

---

# 🧠 Logistic Regression Interpretation

The Logistic Regression model provided useful interpretability.

Some of the strongest positive coefficients were:

```text
social_media_ratio        +3.210501
social_media_hours        +0.809838
gaming_hours              +0.670676
entertainment_ratio       +0.551429
work_study_hours          +0.543946
daily_screen_time_hours   +0.351288
weekend_screen_time       +0.304423
```

Some of the strongest negative coefficients were:

```text
work_study_ratio          -5.926840
gaming_ratio              -5.383891
screen_time_waking_ratio  -1.891359
```

These coefficients should be interpreted as associations learned by the model rather than causal relationships.

---

# 🧪 Final Prediction

The final model was trained for the full 2000 iterations and generated predictions for the test set.

Test prediction shape:

```text
(296302,)
```

Prediction range:

```text
Minimum: 0.0001315475742815282
Maximum: 0.9999999088152761
```

The resulting submission contained:

```text
296,302 rows
2 columns
```

Expected submission columns:

```text
id
addicted_label
```

---

# ✅ Submission Validation

Before saving the submission, the following checks were performed.

```text
Rows: 296302
Duplicate IDs: 0
Missing IDs: 0
Missing predictions: 0
Prediction min: 0.0001315475742815282
Prediction max: 0.9999999088152761
IDs match test: True
```

The first few predictions looked like:

```text
        id  addicted_label
0   691369        0.999388
1   691370        0.931489
2   691371        0.964222
3   691372        0.991131
4   691373        0.995558
```

---

# 📁 Submission Format

The final submission must contain exactly:

```text
id,addicted_label
```

The `addicted_label` column should contain the predicted probability of class `1`.

Example:

```csv
id,addicted_label
691369,0.999388
691370,0.931489
691371,0.964222
691372,0.991131
691373,0.995558
```

Do not convert these probabilities into hard 0/1 predictions because the competition uses ROC-AUC.

---

# 🗂️ Suggested Project Structure

```text
phone-addiction-prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   ├── preprocess.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   └── preprocessing.py
│
├── submissions/
│   └── submission_catboost_d10_2000.csv
│
├── README.md
└── requirements.txt
```

---

# 🛠️ Main Technologies

The project uses:

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- Jupyter Notebook

---

# 📦 Important Libraries

Example environment:

```text
Python 3.x
pandas
numpy
scikit-learn
catboost==1.2.10
jupyter
```

---

# 🚀 Modeling Workflow

The complete workflow is:

```text
Raw Data
   ↓
Load Train/Test
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Missing Value Analysis
   ↓
Train/Validation Split
   ↓
Preprocessing
   ↓
Logistic Regression Baseline
   ↓
CatBoost Experiments
   ↓
Tune Tree Depth
   ↓
Tune Number of Iterations
   ↓
Select Best Model
   ↓
Train Final Model
   ↓
Predict Test Probabilities
   ↓
Validate Submission
   ↓
Save CSV
   ↓
Submit to Competition
```

---

# 📊 Experiment Summary

The progression of the model was:

```text
Logistic Regression
ROC-AUC = 0.917445
```

Then:

```text
CatBoost
ROC-AUC = 0.959040
```

Then depth experiments:

```text
Depth 6 → 0.957701
Depth 8 → 0.959760
Depth 9 → 0.960385
Depth 10 → 0.960802
```

Increasing iterations to 2000 produced:

```text
Depth 10 + 2000 iterations
ROC-AUC = 0.962327
```

Therefore the current best model is:

```text
CatBoost
depth=10
iterations=2000
```

---

# ⚠️ Important Notes

## 1. Do not use hard labels

Because the competition uses ROC-AUC, submit:

```text
0.87342
0.91231
0.10421
```

rather than:

```text
1
1
0
```

---

## 2. Keep test IDs unchanged

The test IDs must remain exactly aligned with the original test data.

The final validation confirmed:

```text
IDs match test: True
```

---

## 3. Missing values are expected

The dataset naturally contains many missing values.

Rows should not be removed simply because some fields are missing.

The preprocessing pipeline handles missing values.

---

## 4. Feature engineering is important

The engineered ratios and behavioral features improved the model's ability to capture relationships between phone usage variables.

---

# 🧾 Final Result

### Best validation ROC-AUC

```text
0.962327
```

### Best model

```text
CatBoost
```

### Configuration

```text
depth = 10
iterations = 2000
```

### Test rows predicted

```text
296,302
```

### Submission columns

```text
id
addicted_label
```

### Submission validation

```text
Rows:               296302
Duplicate IDs:      0
Missing IDs:        0
Missing predictions: 0
IDs match test:     True
```

---

# 🎯 Conclusion

This project demonstrates a complete tabular machine-learning workflow for phone addiction prediction.

The key improvements came from:

1. Behavioral feature engineering
2. Explicit missing-value handling
3. A stratified train/validation split
4. Establishing a Logistic Regression baseline
5. Switching to CatBoost for nonlinear relationships
6. Tuning CatBoost tree depth
7. Increasing iterations from 1000 to 2000
8. Producing probability-based predictions for ROC-AUC evaluation

The final validation ROC-AUC reached:

```text
0.962327
```

with the CatBoost depth-10, 2000-iteration model.

---

## 👤 Project Status

**Model development:** Complete  
**Feature engineering:** Complete  
**Preprocessing:** Complete  
**Model comparison:** Complete  
**CatBoost tuning:** Complete  
**Final test prediction:** Complete  
**Submission validation:** Complete  

The next step is to submit the generated CSV to the competition and record the resulting public leaderboard score.
