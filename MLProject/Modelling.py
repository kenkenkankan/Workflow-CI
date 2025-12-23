import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    mlflow.autolog()

    # Dataset HASIL preprocessing (numerik semua)
    df = pd.read_csv(
        "MLProject/telco_customer_churn_clean.csv"
    )

    # Target
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    with mlflow.start_run(run_name="CI_Autolog_Model"):
        model.fit(X_train, y_train)
        model.score(X_test, y_test)


if __name__ == "__main__":
    main()
