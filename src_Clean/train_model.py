
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import mlflow

def train_model(model, run_name, params, X_train, y_train, X_test, y_test):

    print(f"Starting run: {run_name}")
    with mlflow.start_run(run_name=run_name) as run:
        # Log Params
        mlflow.log_params(params)
        
        # Log Preprocessor (logged for every run so each run is self-contained)
        mlflow.log_artifact("preprocessor.pkl")

        # Train
        model.fit(X_train, y_train)
        
        # Log Model
        pred_train = model.predict(X_train)
        signature = mlflow.models.infer_signature(X_train, pred_train)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train
        )

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        
        # Confusion Matrix
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model.classes_)
        disp.plot()
        plt.title(f"Confusion Matrix: {run_name}")
        
        filename = f"confusion_matrix_{run_name}.png"
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close() 
        return run.info.run_id
        
        