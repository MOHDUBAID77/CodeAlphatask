import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)

if not os.path.exists("credit_data.csv"):
    import generate_dataset

df = pd.read_csv("credit_data.csv")

# Feature engineering
df["debt_to_income"] = df["debt"] / df["income"]
df["payment_success_rate"] = 1 - (
    df["late_payments"] / df["total_payments"].clip(lower=1)
)
df["available_savings_ratio"] = df["savings"] / df["income"]

X = df.drop("creditworthy", axis=1)
y = df["creditworthy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    )
}

results = {}
roc_data = {}

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob)
    }

    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_data[name] = (fpr, tpr)

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)
    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

# Metrics comparison
results_df = pd.DataFrame(results).T
print("\nModel comparison:")
print(results_df)

results_df.to_csv("outputs/model_comparison.csv")

# ROC curve
plt.figure(figsize=(8, 6))
for name, (fpr, tpr) in roc_data.items():
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], "--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Credit Scoring Models")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curves.png")
plt.close()

# Save Random Forest as the main model
final_model = models["Random Forest"]
joblib.dump(
    {"model": final_model, "features": list(X.columns)},
    "models/credit_scoring_model.pkl"
)

print("\nSaved model: models/credit_scoring_model.pkl")
