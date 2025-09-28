# Kaiburr_DataScience
# Consumer Complaint Text Classification

Automatically classify consumer complaints into four categories:

| Label | Category |
|-------|----------|
| 0     | Credit reporting, repair, or other |
| 1     | Debt collection |
| 2     | Consumer Loan |
| 3     | Mortgage |

---

## Project Overview

This project builds a machine learning pipeline to categorize complaints from the [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database).It uses text data to predict the complaint category efficiently.

**Pipeline Steps:**
1. **Exploratory Data Analysis (EDA) & Feature Engineering**
   - Load dataset, filter relevant categories, handle missing values.
   - Map categories to numeric labels.
2. **Text Preprocessing**
   - Lowercasing, punctuation & number removal.
   - Stopwords removal and lemmatization.
3. **Feature Extraction**
   - Convert text to TF-IDF vectors.
4. **Model Selection & Training**
   - Logistic Regression(the best model), Multinomial Naive Bayes.
   - Evaluate model performance and select the best.
5. **Model Evaluation**
   - Accuracy, precision, recall, F1-score, and confusion matrix.
6. **Prediction**
   - Classify new complaints with the trained model.

---

## Getting Started

### Requirements
- Python 3.9+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `scikit-learn`
- runs in jupyter notebook.

### Dataset
Download and unzip the dataset:


