from sklearn.metrics import accuracy_score

def evaluate(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")