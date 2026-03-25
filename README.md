# Spam-Ham Classifier

A machine learning project for classifying text messages as spam or ham (legitimate), built using Python and scikit-learn.

---

## Overview

This project implements a binary text classification pipeline that identifies whether a given SMS or email message is spam or ham. It covers the full ML workflow: data cleaning, preprocessing, feature extraction, model training, and evaluation.

---

## Project Structure

```
Spam-Ham/
├── cleandata.ipynb     # Data cleaning and preprocessing notebook
└── README.md
```

---

## Workflow

**1. Data Cleaning (`cleandata.ipynb`)**
- Loads the raw SMS/email dataset
- Removes duplicates and null values
- Normalizes labels (spam / ham)
- Applies text preprocessing: lowercasing, punctuation removal, stopword removal, and stemming/lemmatization
- Exports a cleaned dataset ready for model training

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib / seaborn (for visualizations)

Install dependencies with:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Ayusha0011/Spam-Ham.git
cd Spam-Ham
```

2. Open and run the notebook:

```bash
jupyter notebook cleandata.ipynb
```

---

## Dataset

The project uses the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) from the UCI Machine Learning Repository, which contains 5,574 labeled SMS messages.

---

## Results

The cleaned dataset serves as the foundation for classification. Common models applied to this type of dataset (Naive Bayes, Logistic Regression, SVM) typically achieve 97-99% accuracy on held-out test data.

---
