import xgboost as xgb
from sklearn.model_selection import train_test_split

def XGBModel(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    return predictions, y_test