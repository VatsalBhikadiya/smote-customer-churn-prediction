"""
SMOTE for Customer Churn Prediction
Dataset expected: telecom_customer_churn.csv
Install: pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
Run: python smote_churn_prediction.py

Important:
- The dataset is split before SMOTE.
- SMOTE is applied only to the training data to prevent test-data leakage.
- The script compares each classifier without SMOTE and with SMOTE.
- Actual scores depend on the supplied CSV and the installed library versions.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_FILE = Path("telecom_customer_churn.csv")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data():
    """Load the CSV and print basic dataset information."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE}. Put telecom_customer_churn.csv "
            "in the same folder as this script."
        )
    df = pd.read_csv(DATA_FILE)
    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    print("\\nFirst five rows:")
    print(df.head())
    print("\\nMissing values (top 15):")
    print(df.isna().sum().sort_values(ascending=False).head(15))
    return df


def prepare_target(df):
    """Return feature table X, binary target y, and target column name."""
    df = df.copy()
    if "Customer Status" in df.columns:
        status = df["Customer Status"].astype(str).str.strip().str.lower()
        valid = status.isin(["churned", "stayed", "joined"])
        df = df.loc[valid].copy()
        y = status.loc[valid].eq("churned").astype(int)
        target_name = "Customer Status"
    elif "Churn" in df.columns:
        raw = df["Churn"].astype(str).str.strip().str.lower()
        mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
        y = raw.map(mapping)
        valid = y.notna()
        df = df.loc[valid].copy()
        y = y.loc[valid].astype(int)
        target_name = "Churn"
    else:
        raise ValueError("Target column not found. Expected 'Customer Status' or 'Churn'.")

    if y.nunique() != 2:
        raise ValueError("Binary churn modeling requires both target classes.")
    # Exclude target and post-outcome fields to avoid target leakage.
    drop_cols = [target_name, "Churn", "Churn Category", "Churn Reason"]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Identifiers are not generalizable predictive features.
    id_cols = [
        c for c in X.columns
        if c.strip().lower() in {"customerid", "customer id", "id"}
    ]
    X = X.drop(columns=id_cols, errors="ignore")
    return X, y, target_name


def make_preprocessor(X):
    """Build numeric and categorical preprocessing pipelines."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])


def save_class_distribution(y, filename, title):
    """Save a bar chart for the supplied target labels."""
    labels = y.map({0: "Not Churned", 1: "Churned"})
    counts = labels.value_counts().reindex(["Not Churned", "Churned"], fill_value=0)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(title)
    plt.xlabel("Customer class")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y, target_name = prepare_target(df)

    print("\\nTarget column:", target_name)
    print("Class distribution (0 = Not Churned, 1 = Churned):")
    print(y.value_counts().sort_index().rename(index={0: "Not Churned", 1: "Churned"}))

    save_class_distribution(y, "churn_distribution_original.png",
                            "Original Customer Churn Distribution")

    # Split first. Test data remains untouched by SMOTE.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("\\nTraining class counts before SMOTE:")
    print(y_train.value_counts().sort_index())

    preprocessor = make_preprocessor(X)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1
        )
    }

    results = []
    fitted = {}
    for model_name, estimator in models.items():
        for sampling in ["Without SMOTE", "With SMOTE"]:
            steps = [("preprocessor", preprocessor)]
            if sampling == "With SMOTE":
                steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)))
            steps.append(("model", estimator))
            # imblearn Pipeline applies SMOTE only during fit, not prediction.
            pipe = ImbPipeline(steps)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            probabilities = pipe.predict_proba(X_test)[:, 1]

            row = {
                "Model": model_name,
                "Sampling": sampling,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 Score": f1_score(y_test, pred, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, probabilities)
            }
            results.append(row)
            fitted[(model_name, sampling)] = pipe
            print(f"\\n{model_name} — {sampling}")
            print(classification_report(
                y_test, pred, labels=[0, 1],
                target_names=["Not Churned", "Churned"], zero_division=0
            ))
            cm = confusion_matrix(y_test, pred, labels=[0, 1])
            plt.figure(figsize=(5.5, 4.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Not Churned", "Churned"],
                        yticklabels=["Not Churned", "Churned"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"{model_name}\\n{sampling}")
            safe_name = model_name.lower().replace(" ", "_")
            safe_sampling = sampling.lower().replace(" ", "_")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"confusion_{safe_name}_{safe_sampling}.png", dpi=200)
            plt.close()

    metrics = pd.DataFrame(results)
    metrics.to_csv(OUTPUT_DIR / "smote_model_metrics.csv", index=False)
    print("\\nModel performance comparison:")
    print(metrics.to_string(index=False, formatters={
        c: (lambda v: f"{v:.4f}") for c in
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    }))

    # Comparison plot for each metric and model.
    plot_data = metrics.melt(
        id_vars=["Model", "Sampling"],
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        var_name="Metric", value_name="Score"
    )
    g = sns.catplot(
        data=plot_data, x="Model", y="Score", hue="Sampling",
        col="Metric", col_wrap=3, kind="bar", height=4, aspect=1.15,
        sharey=True
    )
    g.set(ylim=(0, 1))
    g.set_xticklabels(rotation=20)
    g.fig.suptitle("Model Performance: Without SMOTE vs With SMOTE", y=1.02)
    g.tight_layout()
    g.savefig(OUTPUT_DIR / "model_performance_comparison.png", dpi=200)
    plt.close("all")

    # Summarize class counts before/after resampling on the training set.
    transformed = preprocessor.fit_transform(X_train)
    X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(
        transformed, y_train
    )
    before = y_train.value_counts().reindex([0, 1], fill_value=0)
    after = pd.Series(y_resampled).value_counts().reindex([0, 1], fill_value=0)
    balance = pd.DataFrame({
        "Class": ["Not Churned", "Churned"],
        "Before SMOTE": [int(before[0]), int(before[1])],
        "After SMOTE": [int(after[0]), int(after[1])]
    })
    balance.to_csv(OUTPUT_DIR / "smote_training_class_counts.csv", index=False)
    balance_plot = balance.set_index("Class").plot(kind="bar", figsize=(7, 5))
    plt.title("Training Data Class Balance Before and After SMOTE")
    plt.xlabel("Customer class")
    plt.ylabel("Number of training records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "smote_class_balance.png", dpi=200)
    plt.close()

    best = metrics.sort_values("F1 Score", ascending=False).iloc[0]
    print("\\nBest result by F1 Score:")
    print(best.to_string())
    print("\\nOutput files saved in:", OUTPUT_DIR.resolve())
    print("Note: Results are specific to this dataset split and are not guaranteed future performance.")


if __name__ == "__main__":
    main()
