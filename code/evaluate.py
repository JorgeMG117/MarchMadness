import pandas as pd
import numpy as np


def evaluate(results):
    num_games = results['Wins'].sum()
    print(f"Number of games: {num_games}")#63

    assert results['Wins'].sum() == results['PredictedWins'].sum()
    
    print(results.head())
    """
        TeamID      TeamName  PredictedWins  Wins
    0    1163   Connecticut     6              6
    1    1235       Iowa St     2              1
    2    1228      Illinois     3              3
    3    1120        Auburn     0              2
    4    1361  San Diego St     2              1
    """
    results['Diff'] = results['PredictedWins'] - results['Wins']
    results['AbsDiff'] = results['Diff'].abs()
    errors = results['AbsDiff'].sum()
    print(f"Errors: {errors}")#90

    results['Accuracy'] = np.where(results['Diff'] <= 0, 
                             results['PredictedWins'], 
                             results['Wins'])

    accuracy = results['Accuracy'].sum() / num_games

    results['Points'] = results['Accuracy'].apply(lambda x: 10 * 2 ** (x - 1) if x != 0 else 0)
    
    points = results['Points'].sum()
    
    return accuracy, points


def compare_bracket(predictions):
    actual_results = pd.read_csv('predictions/results.csv')

    predictions_df = pd.DataFrame(list(predictions.items()), columns=['TeamID', 'PredictedWins'])

    # Merge with the actual wins DataFrame
    merged_df = actual_results.merge(predictions_df, on='TeamID')

    accuracy, points = evaluate(merged_df)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Points: {points}")

    # Compare against analyst
    pro_pred = ['predictions/stephen_a.csv', 'predictions/shae_cornette.csv']
    for pred_file in pro_pred:
        pro_pred_df = pd.read_csv(pred_file)
        pro_pred_df.rename(columns={'Wins': 'PredictedWins'}, inplace=True)
        pro_pred_df = pro_pred_df.merge(actual_results, on=['TeamID', 'TeamName'])

        accuracy, points = evaluate(pro_pred_df)
        print(f"Accuracy for {pred_file}: {accuracy:.2f}")
        print(f"Points for {pred_file}: {points}")
        

    # si lo he predicho todo bien, la resta entre actual y predicho es 0
