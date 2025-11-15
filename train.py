import os

import dagshub
dagshub.init(repo_owner='CodingAnas', repo_name='student-result-prediction', mlflow=True)

import pandas as pd
df = pd.read_csv("./data/student_scores.csv")

X = df[["study_hours", "sleep_hours"]]
y = df["scores"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(
    X_train,
    y_train
)

import mlflow
with mlflow.start_run():

    mlflow.log_param("num_train_samples", len(X_train))
    mlflow.log_param("num_test_samples", len(X_test))
    mlflow.log_param("features", list(X.columns))

    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mean_absolute_error", MAE)
    mlflow.log_metric("mean_squared_error", MSE)
    mlflow.log_metric("r2_score", r2)

    mlflow.sklearn.log_model(model, "model")

    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "./models/model1.pkl")