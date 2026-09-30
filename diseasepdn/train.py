import os, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from config import DATA_PATH, TARGET_COLUMN, DROP_COLUMNS, TEST_SIZE, RANDOM_STATE

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found. Columns: {list(df.columns)}")
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).dropna(subset=[TARGET_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    classes = list(pd.Series(y).unique())
    mapping = {str(c): i for i, c in enumerate(classes)}
    y = y.map(lambda v: mapping[str(v)])

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, eval_metric="logloss"
        )
    }

    os.makedirs("models", exist_ok=True)
    results = []

    for name, model in models.items():
        pipe = Pipeline([("preprocessor", pre), ("model", model)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results.append({
            "model": name,
            "accuracy": accuracy_score(yte, pred),
            "precision_weighted": precision_score(yte, pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(yte, pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0)
        })
        filename = name.lower().replace(" ", "_")
        joblib.dump({"pipeline": pipe, "target_classes": classes, "feature_columns": list(X.columns)},
                    f"models/{filename}.joblib")
        print("\n" + "="*60 + f"\n{name}\n" + "="*60)
        print(classification_report(yte, pred, zero_division=0))

    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()
