# Netflix Retention Modeling
> Predicting user churn from behavioral patterns using Logistic Regression and Random Forest

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## Overview

Streaming platforms often detect churn only **after** engagement has already declined.  
This project builds a **behavioral churn prediction model** using Netflix watch session data to identify users who are likely to stop using the platform.

By transforming **session-level watch activity into user-level behavioral features**, we train machine learning models to predict churn risk.

### Key Questions
- Which engagement signals best predict churn?
- How early can we detect declining retention?
- Do simple models (Logistic Regression) perform competitively with ensemble methods (Random Forest)?

---

## Project Structure

```
Netflix-Retention-Modeling
│
├── data
│   ├── watch_history.csv
│   ├── movies.csv
│   ├── watch_joined.csv
│   ├── watch_preprocessed.csv
│   └── user_features.csv
│
├── outputs
│   ├── model_performance.csv
│   ├── roc_curve_comparison.png
│   ├── best_model_confusion_matrix.png
│   └── test_predictions.csv
│
├── src
│   ├── 00_data_preparation.py
│   ├── 01_eda.py
│   ├── 02_preprocessing.py
│   ├── 03_feature_engineering.py
│   └── 04_modeling.py
│
└── README.md
```

## Dataset

**Source:** Kaggle — Netflix User Watch History

| Property | Details |
|----------|---------|
| Watch Sessions | ~105,000 |
| Users | ~10,000 |
| Time Period | Jan 2024 – Dec 2025 |

### Key Columns

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `watch_date` | Date of watch session |
| `watch_duration_minutes` | Minutes watched |
| `completion_rate` | Percentage of content watched |
| `action` | Indicates the viewing outcome of a session |
| `genre_primary` | Primary content genre |
| `content_type` | Movie vs TV Series |
| `device_type` | Device used for viewing |

---

## Methodology

```
Raw Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation
```

---

### 1. Exploratory Data Analysis

EDA was conducted to understand data quality and user engagement patterns.

- Missing value analysis
- Watch duration distribution
- Genre and device usage distribution
- Session activity per user
- Initial churn ratio exploration

---

### 2. Data Preprocessing

Columns removed due to redundancy or poor data quality:

| Column | Reason |
|--------|--------|
| `progress_percentage` | Duplicate of `completion_rate` |
| `watch_ratio` | Derived column |
| `user_rating` | 79.9% missing values |
| `genre_secondary` | 64% missing values |
| `session_id` | Identifier only |

**Outlier Removal** — Watch duration anomalies filtered using:
```python
watch_duration_minutes / duration_minutes < 3
```

**Missing Values** — Replaced using median imputation.

---

### 3. Churn Definition

A user is labeled **churned** if they have **no watch activity within the last 30 days** of the dataset.

```python
reference_date = df['watch_date'].max()
last_watch = df.groupby('user_id')['watch_date'].max()
recency_days = (reference_date - last_watch).dt.days
churned = (recency_days >= 30).astype(int)
```

| Label | Share |
|-------|-------|
| Churned | ~65% |
| Retained | ~35% |

---
## 4.1 Data Description

This project uses session-level Netflix viewing data and transforms it into user-level behavioral features for churn prediction.

### Core Raw Columns

| Column | Description | Why it matters |
|--------|-------------|----------------|
| `user_id` | Unique user identifier | Used to aggregate session-level data into user-level features |
| `watch_date` | Date of each viewing session | Used to calculate recency and activity patterns |
| `watch_duration_minutes` | Minutes watched per session | Measures overall engagement intensity |
| `completion_rate` | Percentage of content watched | Captures how fully users consume content |
| `action` | Session outcome (e.g., completed) | Used to compute `completion_ratio`, a key engagement metric |
| `genre_primary` | Primary genre of content | Used to measure content diversity |
| `content_type` | Movie or TV Series | Used to compute viewing preference ratios |
| `device_type` | Device used for viewing | Used to measure cross-device engagement |
| `is_netflix_original` | Whether content is Netflix original | Used to capture preference for platform-owned content |

## 4.2 Feature Engineering

Raw session-level data was aggregated into user-level features to capture long-term engagement behavior.

### Engineered Features

| Feature | How it was created | Why it matters |
|--------|------------------|----------------|
| `session_frequency` | Total sessions / active days | Measures consistency of user engagement |
| `completion_ratio` | Completed sessions / total sessions | Captures how fully users consume content |
| `recency_days` | Days since last activity | Strong indicator of churn risk |
| `genre_diversity` | Number of unique genres watched | Reflects breadth of content preference |
| `avg_watch_time` | Average watch duration per session | Indicates engagement intensity |
| `movie_ratio` | Movie sessions / total sessions | Captures content type preference |
| `original_ratio` | Netflix original sessions / total sessions | Measures platform-specific preference |
| `device_diversity` | Number of unique devices used | Indicates cross-device engagement |

---

### 5. Modeling

Three models were trained and compared:

- Baseline (Most Frequent)
- Logistic Regression
- Random Forest

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|------|------|------|------|------|------|------|
| Logistic Regression | 0.668 | **0.859** | 0.604 | 0.709 | **0.802** | **0.870** |
| Random Forest | **0.762** | 0.783 | **0.891** | **0.833** | 0.780 | 0.847 |
| Baseline | 0.669 | 0.669 | 1.000 | 0.802 | 0.500 | 0.669 |

**Best Model:** Logistic Regression  
- ROC-AUC = **0.802**
- PR-AUC = **0.870**
- F1 Score = **0.709**

### ROC Curve  
<p align="center">
  <img src="outputs/roc_curve_comparison.png" width="500">
</p>

**What this shows:**  
The ROC curve compares the model’s ability to distinguish between churn and non-churn users across different thresholds.

**Key insight:**  
The Random Forest model shows strong classification performance with a high ROC-AUC, indicating effective separation between churn and active users.

### Confusion Matrix  
<p align="center">
  <img src="outputs/best_model_confusion_matrix.png" width="500">
</p>

**What this shows:**  
The confusion matrix summarizes the model’s predictions by comparing actual vs predicted churn outcomes.

**Key insight:**  
The model correctly identifies a large portion of churn users (high recall), but some non-churn users are misclassified, suggesting a trade-off between precision and recall.>
---

## Key Insights

- User engagement patterns strongly predict churn risk.
- Session frequency and completion behavior are important indicators of retention.
- Logistic Regression achieved the best ranking performance (ROC-AUC).
- Random Forest achieved the highest recall and F1 score.

## Key Findings (Preliminary)

- **Session frequency** is the strongest early signal of churn risk
- Users with declining `completion_ratio` are more likely to churn
- **Genre diversity** tends to increase long-term retention
- Longer inactivity (`recency_days`) strongly indicates churn risk
---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/netflix-retention-modeling.git
cd netflix-retention-modeling
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the pipeline in order**
```bash
python scripts/00_data_preparation.py
python scripts/01_eda.py
python scripts/02_preprocessing.py
python scripts/03_feature_engineering.py
python scripts/04_modeling.py
```
