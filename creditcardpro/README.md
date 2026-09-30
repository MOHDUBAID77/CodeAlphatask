# Credit Scoring ML Project

A beginner-friendly machine learning project that predicts creditworthiness using financial history.

## Models
- Logistic Regression
- Decision Tree
- Random Forest

## Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

## Dataset
The project generates a synthetic financial dataset automatically, so no external download is required.

## How to run

```bash
pip install -r requirements.txt
python train_model.py
```

The trained Random Forest model is saved as `models/credit_scoring_model.pkl`.

To run the interactive prediction app:

```bash
streamlit run app.py
```
