"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import mlflow
import joblib
### Import MLflow

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    joblib.dump(col_transf, 'preprocessor.pkl')
    # Log the transformer as an artifact
    mlflow.log_artifact("./preprocessor.pkl")
    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    predictions = log_reg.predict(X_train)

    # Infer signature (input and output schema)
    signature = mlflow.models.infer_signature(X_train, predictions)
    # Log model
    mlflow.sklearn.log_model(
        sk_model=log_reg,
        artifact_path="log_model",
        signature=signature,
        input_example=X_train
    )
    ### Log the data
    train_df = X_train.copy()
    train_df['Exited'] = y_train  # Add target back
    dataset = mlflow.data.from_pandas(train_df, targets='Exited')
    mlflow.log_input(dataset, context="training")
    return log_reg

def trainNew(model,run_name,params,X_train,y_train,X_test,y_test):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        model.fit(X_train,y_train)
        pred=model.predict(X_train)
        signature = mlflow.models.infer_signature(X_train, pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{run_name}",
            signature=signature,
            input_example=X_train
        )

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
    )
        plt.title(f"Confusion Matrix: {run_name}")
        plt.savefig(f"confusion_matrix_{run_name}.png")
        mlflow.log_artifact(f"confusion_matrix_{run_name}.png")
        conf_mat_disp.plot()
            
    
        plt.close()



def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    ### Set the experiment name
    mlflow.set_experiment("Log Experiment")
    exp = mlflow.get_experiment_by_name("Log Experiment")
    exp_id = exp.experiment_id
    max_iter = 1000
    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run(experiment_id=exp_id, run_name="LogisticRegression"):

        df = pd.read_csv("../dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    ### Log the max_iter parameter
        mlflow.log_param("max_iter", max_iter)

        model = train(X_train, y_train)

    
        y_pred = model.predict(X_test)

    ### Log metrics after calculating them
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("precision",precision)
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("f1",f1)

    ### Log tag
        mlflow.set_tag("version", "1.0")
        mlflow.set_tag("model", "LogisticRegression")
 
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_)
        conf_mat_disp.plot()
    mlflow.end_run()

    rf=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
    rf_params = {"n_estimators": 100, "max_depth": 10}
    trainNew(model=rf,
             run_name="RandomForest",
             params=rf_params,
             X_train=X_train,
               y_train=y_train,
        X_test=X_test, 
        y_test=y_test)


    gb=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,random_state=42)
    gb_params = {"n_estimators": 100, "learning_rate": 0.1}
    trainNew(
        model=gb,
        run_name="GradientBoosting",
        params=gb_params,
             X_train=X_train,
               y_train=y_train,
        X_test=X_test, 
        y_test=y_test)
    mlflow.set_tracking_uri("http://localhost:5000")
    run_rf = "3690f8b291404a2cbbb18de2ba6846e9" 
    run_lr = "4d758d580a204ac4b0f1bfe06bb86067"

    mlflow.register_model(f"runs:/{run_lr}/log_model", "logistic_Model")
    mlflow.register_model(f"runs:/{run_rf}/RandomForest", "randomForst_Model")


if __name__ == "__main__":
    main()
