from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.metrics import precision_score


def evaluate(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")


def compare_bracket(predictions):
    actual_results = pd.read_csv('predictions/results.csv')

    predictions_df = pd.DataFrame(list(predictions.items()), columns=['TeamID', 'PredictedWins'])

    # Merge with the actual wins DataFrame
    merged_df = actual_results.merge(predictions_df, on='TeamID')
    print(merged_df.head())

    # Number of games
    num_games =  merged_df['Wins'].sum()
    print(num_games)

    merged_df['AbsDiff'] = (merged_df['PredictedWins'] - merged_df['Wins']).abs()
    errors = merged_df['AbsDiff'].sum()

    accuracy = (num_games - errors) / num_games

    print(f"Accuracy: {accuracy:.2f}")

    # Compare against analyst
    pro_pred = ['predictions/stephen_a.csv']
    for pred_file in pro_pred:
        pro_pred_df = pd.read_csv(pred_file)
        pro_pred_df.rename(columns={'Wins': 'PredictedWins'}, inplace=True)
        pro_pred_df = pro_pred_df.merge(actual_results, on=['TeamID', 'TeamName'])

        num_games = pro_pred_df['Wins'].sum()
        print(f"Number of games: {num_games}")
       
        print(pro_pred_df.head())
        pro_pred_df['AbsDiff'] = (pro_pred_df['PredictedWins'] - pro_pred_df['Wins']).abs()
        errors = pro_pred_df['AbsDiff'].sum()

        accuracy = (num_games - errors) / num_games
        print(f"Accuracy for {pred_file}: {accuracy:.2f}")
        

    # si lo he predicho todo bien, la resta entre actual y predicho es 0
