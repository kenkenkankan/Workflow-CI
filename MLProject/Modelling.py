import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def main():
    # WAJIB: autolog untuk Kriteria 3 (konsisten dgn Kriteria 2)
    mlflow.autolog()

    # Pastikan path sesuai struktur MLProject
    df = pd.read_csv("telco_customer_churn_clean.csv")

    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TANPA manual logging
    with mlflow.start_run(run_name="CI_Autolog_Model"):
        model.fit(X_train, y_train)
        model.score(X_test, y_test)


if __name__ == "__main__":
    main()
