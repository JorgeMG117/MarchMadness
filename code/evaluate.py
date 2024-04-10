from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.metrics import precision_score


def evaluate(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

def calculate_metrics(merged_df):
    # Assuming 'merged_df' contains 'Wins' and 'PredictedWins' for each team
    
    # Calculate max number of wins for setting up the binary classification for each win
    max_wins = max(merged_df['Wins'].max(), merged_df['PredictedWins'].max())

    # Initialize counters
    TP = FP = TN = FN = 0

    # For each team, check each win up to the max_wins
    for _, row in merged_df.iterrows():
        for win in range(1, max_wins + 1):
            actual_win = win <= row['Wins']
            predicted_win = win <= row['PredictedWins']
            
            if actual_win and predicted_win:
                TP += 1
            elif not actual_win and predicted_win:
                FP += 1
            elif not actual_win and not predicted_win:
                TN += 1
            elif actual_win and not predicted_win:
                FN += 1

    # Calculate metrics
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, accuracy, f1

def compare_bracket(predictions):
    actual_results = pd.read_csv('predictions/results.csv')

    predictions_df = pd.DataFrame(list(predictions.items()), columns=['TeamID', 'PredictedWins'])

    # Merge with the actual wins DataFrame
    merged_df = actual_results.merge(predictions_df, on='TeamID')

    # Number of games
    num_games =  merged_df['Wins'].sum()
    print(num_games)
    num_games = 67

    errors = 0
    for _, row in merged_df.iterrows():
        actual_wins = row['Wins']
        predicted_wins = row['PredictedWins']
        
        # Compute diference
        diff = abs(actual_wins - predicted_wins)

        errors += diff

    accuracy = (num_games - errors) / num_games

    print(f"Accuracy: {accuracy:.2f}")
        
        

    # si lo he predicho todo bien, la resta entre actual y predicho es 0
