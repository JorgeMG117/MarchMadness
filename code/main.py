from neural_network import NNPredictor
from dataset import NCAADataset
from xgb import XGBPredictor
from evaluate import evaluate
from evaluate import compare_bracket
import numpy as np
import pandas as pd


def create_matchups(df):
    # Specifying the order for a playoff structure
    order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

    # Adjusting the order (subtracting 1 for zero-based indexing)
    ordered_df = df.iloc[[o-1 for o in order]].reset_index(drop=True)

    return ordered_df


def simulate_matchups(df, expand_matchup, predict_matchup, predictions):
    #print(df.head())
    winners = []
    for i in range(0, len(df), 2):
        team1 = df.iloc[i]
        team2 = df.iloc[i+1]

        print(f"Matchup {i//2 + 1}: {team1['TeamName']} vs {team2['TeamName']}")

        # Prepare data to make prediction
        merged_df = pd.DataFrame(columns=['T1_TeamID', 'T2_TeamID'])
        new_row = pd.DataFrame({'T1_TeamID': [team1['TeamID']], 'T2_TeamID': [team2['TeamID']], 'Season': [2024]})
        merged_df = pd.concat([merged_df, new_row], ignore_index=True)
        merged_df = expand_matchup(merged_df)
        merged_df = merged_df.drop(columns=['T1_TeamID', 'T2_TeamID', 'Season'])
        #print(merged_df.head())
        #print(merged_df.shape)
        
        # Predict winner
        winner = predict_matchup(merged_df)
        if winner:
            winner_team = team1
            loser_team = team2
        else:
            winner_team = team2
            loser_team = team1

        # Increment win count in predictions
        predictions[winner_team['TeamID']] = predictions.get(winner_team['TeamID'], 0) + 1
        predictions[loser_team['TeamID']] = predictions.get(loser_team['TeamID'], 0)

        winners.append(winner_team)
        print(f"Winner: {winners[-1]['TeamName']}\n")

    winners_df = pd.DataFrame(winners).reset_index(drop=True)
    return winners_df


def simulate_bracket(bracket_data, expand_matchup, predict_matchup, predictions):
    # W/X/Y/Z are East,Midwest,South,West. TODO Ojo con esto, creo que esta mal

    # Order vector so that it follows the bracket structure
    matchups_w = create_matchups(bracket_data[bracket_data['Seed'].str.startswith('W')].sort_values(by='Seed'))
    matchups_x = create_matchups(bracket_data[bracket_data['Seed'].str.startswith('X')].sort_values(by='Seed'))
    matchups_y = create_matchups(bracket_data[bracket_data['Seed'].str.startswith('Y')].sort_values(by='Seed'))
    matchups_z = create_matchups(bracket_data[bracket_data['Seed'].str.startswith('Z')].sort_values(by='Seed'))
    

    print("### First round ###\n")
    print("# East #")
    matchups_w = simulate_matchups(matchups_w, expand_matchup, predict_matchup, predictions)
    print("# West #")
    matchups_x = simulate_matchups(matchups_x, expand_matchup, predict_matchup, predictions)
    print("# South #")
    matchups_z = simulate_matchups(matchups_z, expand_matchup, predict_matchup, predictions)
    print("# Midwest #")
    matchups_y = simulate_matchups(matchups_y, expand_matchup, predict_matchup, predictions)
    
    
    print("### Second round ###")
    print("# East #")
    matchups_w = simulate_matchups(matchups_w, expand_matchup, predict_matchup, predictions)
    print("# West #")
    matchups_x = simulate_matchups(matchups_x, expand_matchup, predict_matchup, predictions)
    print("# South #")
    matchups_z = simulate_matchups(matchups_z, expand_matchup, predict_matchup, predictions)
    print("# Midwest #")
    matchups_y = simulate_matchups(matchups_y, expand_matchup, predict_matchup, predictions)
    
    print("### Sweet 16 ###")
    print("# East #")
    matchups_w = simulate_matchups(matchups_w, expand_matchup, predict_matchup, predictions)
    print("# West #")
    matchups_x = simulate_matchups(matchups_x, expand_matchup, predict_matchup, predictions)
    print("# South #")
    matchups_z = simulate_matchups(matchups_z, expand_matchup, predict_matchup, predictions)
    print("# Midwest #")
    matchups_y = simulate_matchups(matchups_y, expand_matchup, predict_matchup, predictions)
    

    print("### Elite 8 ###")
    print("# East #")
    winner_w = simulate_matchups(matchups_w, expand_matchup, predict_matchup, predictions)
    print("# West #")
    winner_x = simulate_matchups(matchups_x, expand_matchup, predict_matchup, predictions)
    print("# South #")
    winner_z = simulate_matchups(matchups_z, expand_matchup, predict_matchup, predictions)
    print("# Midwest #")
    winner_y = simulate_matchups(matchups_y, expand_matchup, predict_matchup, predictions)
    

    print("### Final Four ###")
    matchups = pd.concat([winner_w, winner_x, winner_z, winner_y], ignore_index=True)
    matchups = simulate_matchups(matchups, expand_matchup, predict_matchup, predictions)

    print("### Championship ###")
    winner = simulate_matchups(matchups, expand_matchup, predict_matchup, predictions)


    print("### Champion ###")
    print(winner['TeamName'].values[0])

    


def main():
    # Create dataset
    dataset = NCAADataset()

    # Data
    print("### DATA ###")
    print(dataset.tournament_data.head())
    print(dataset.tournament_data.tail())
    print(dataset.tournament_data.shape)
    #print(dataset.tournament_data.columns)


    print("### TRAINING MODEL ###")

    xgb = XGBPredictor(dataset.tournament_data)

    xgb.train_model()

    #nn = NNPredictor(dataset.tournament_data)

    #nn.train_model()
    


    # Evaluate the model

    #evaluate(predictions, y_test)


    # Simulate 2024 bracket
    print("### 2024 BRACKET ###")
    bracket_data = dataset.tournament_2024[dataset.tournament_2024['Tournament'] == 'M']
    bracket_data = bracket_data.merge(dataset.teams, on='TeamID', how='inner')
    bracket_data = bracket_data[['Seed', 'TeamName', 'TeamID']]
    #print(bracket_data.head())

    
    predictions = {}
    match_predictor = xgb.predict_matchup
    simulate_bracket(bracket_data, dataset.expand_matchup, match_predictor, predictions)

    # Evaluate predictions with the actual results
    compare_bracket(predictions)

    #dataset.tournament_2024 = dataset.tournament_2024[]


if __name__ == '__main__':
    main()