import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    # WAJIB: autolog (konsisten dgn Kriteria 2)
    mlflow.autolog()

    # Dataset hasil preprocessing Kriteria 1
    df = pd.read_csv("telco_customer_churn_clean.csv")

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    with mlflow.start_run(run_name="CI_Model_No_Preprocessing"):
        model.fit(X_train, y_train)
        model.score(X_test, y_test)


if __name__ == "__main__":
    main()
