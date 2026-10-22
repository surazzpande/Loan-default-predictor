import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from preprocess import clean, add_features

df = add_features(clean(pd.read_csv("data/credit_risk_dataset.csv")))
X = df.drop(columns="loan_status")
y = df["loan_status"]

cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
num_cols = [c for c in X.columns if c not in cat_cols]

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                             scale_pos_weight=(y == 0).sum() / (y == 1).sum()),
}

for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} ===  ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
    print(classification_report(y_test, preds))
    if name == "XGBoost":
        joblib.dump(pipe, "models/model.pkl")