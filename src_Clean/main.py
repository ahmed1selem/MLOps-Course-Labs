import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import make_column_transformer
import mlflow
import mlflow.sklearn
import mlflow.models
import train_model
import data_preprocessing

def register_model(run_id, model_registry_name, artifact_path):
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, model_registry_name)
def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Log Experiment")
    
    # Clean up any "zombie" runs from previous crashes
    if mlflow.active_run():
        print("Closing active zombie run...")
        mlflow.end_run()

    # Load Data (Once)
    try:
        df = pd.read_csv("data/Churn_Modelling.csv")
    except FileNotFoundError:
        df = pd.read_csv("../dataset/Churn_Modelling.csv")

    # Preprocess (Once)
    col_transf, X_train, X_test, y_train, y_test = data_preprocessing(df)

    # === EXPERIMENT 1: Logistic Regression ===
    lr = LogisticRegression(max_iter=1000)
    run_id_lr=train_model(
        model=lr,
        run_name="LogisticRegression",
        params={"max_iter": 1000},
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    # === EXPERIMENT 2: Random Forest ===
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    run_id_rf=train_model(
        model=rf,
        run_name="RandomForest",
        params={"n_estimators": 100, "max_depth": 10},
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    # === EXPERIMENT 3: Gradient Boosting ===
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    run_id_gb=train_model(
        model=gb,
        run_name="GradientBoosting",
        params={"n_estimators": 100, "learning_rate": 0.1},
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    register_model(
        run_id_lr, 
        model_registry_name="logistic_Model", 
        artifact_path="log_model"
    )
    register_model(
        run_id_rf, 
        model_registry_name="randomForst_Model", 
        artifact_path="randomForst"
    )


if __name__ == "__main__":
    main()