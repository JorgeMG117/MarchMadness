from dataset import NCAADataset
from model import XGBModel
from evaluate import evaluate
import numpy as np
import pandas as pd
from model import predict_matchup

def create_matchups(df):
    # Specifying the order for a playoff structure
    order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

    # Adjusting the order (subtracting 1 for zero-based indexing)
    ordered_df = df.iloc[[o-1 for o in order]].reset_index(drop=True)

    return ordered_df


def simulate_matchups(df, expand_matchup):
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
        winner = team1 if np.random.rand() > 0.5 else team2
        winners.append(winner)
        #predict_matchup(merged_df, model)

        #print(f"Winner: {winner['TeamName']}\n")

    winners_df = pd.DataFrame(winners).reset_index(drop=True)
    return winners_df


def simulate_bracket(bracket_data, expand_matchup):
    # W/X/Y/Z are East,Midwest,South,West. TODO Ojo con esto, creo que esta mal

    # Order vector so that it follows the bracket structure
    matchups_w = create_matchups(bracket_data[bracket_data['Seed'].str.startswith('W')].sort_values(by='Seed'))
    matchups_x = create_matchups(bracket_data[bracket_data['Seed'].str.startswith('X')].sort_values(by='Seed'))
    matchups_y = create_matchups(bracket_data[bracket_data['Seed'].str.startswith('Y')].sort_values(by='Seed'))
    matchups_z = create_matchups(bracket_data[bracket_data['Seed'].str.startswith('Z')].sort_values(by='Seed'))
    

    print("### First round ###\n")
    print("# East #")
    matchups_w = simulate_matchups(matchups_w, expand_matchup)
    print("# Midwest #")
    matchups_x = simulate_matchups(matchups_x, expand_matchup)
    print("# South #")
    matchups_y = simulate_matchups(matchups_y, expand_matchup)
    print("# West #")
    matchups_z = simulate_matchups(matchups_z, expand_matchup)

    
    print("### Second round ###")
    print("# East #")
    matchups_w = simulate_matchups(matchups_w, expand_matchup)
    print("# Midwest #")
    matchups_x = simulate_matchups(matchups_x, expand_matchup)
    print("# South #")
    matchups_y = simulate_matchups(matchups_y, expand_matchup)
    print("# West #")
    matchups_z = simulate_matchups(matchups_z, expand_matchup)

    print("### Sweet 16 ###")
    print("# East #")
    matchups_w = simulate_matchups(matchups_w, expand_matchup)
    print("# Midwest #")
    matchups_x = simulate_matchups(matchups_x, expand_matchup)
    print("# South #")
    matchups_y = simulate_matchups(matchups_y, expand_matchup)
    print("# West #")
    matchups_z = simulate_matchups(matchups_z, expand_matchup)
    

    print("### Elite 8 ###")
    print("# East #")
    winner_w = simulate_matchups(matchups_w, expand_matchup)
    print("# Midwest #")
    winner_x = simulate_matchups(matchups_x, expand_matchup)
    print("# South #")
    winner_y = simulate_matchups(matchups_y, expand_matchup)
    print("# West #")
    winner_z = simulate_matchups(matchups_z, expand_matchup)


    print("### Final Four ###")
    matchups = pd.concat([winner_w, winner_x, winner_y, winner_z], ignore_index=True)
    matchups = simulate_matchups(matchups, expand_matchup)

    print("### Championship ###")
    winner = simulate_matchups(matchups, expand_matchup)


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

    y = dataset.tournament_data['T1_Score'] - dataset.tournament_data['T2_Score']
    #y = np.where(y > 0, 1, 0)
    #print(y.head())
    #print(y.shape)
    
    X = dataset.tournament_data[dataset.tournament_data.columns[6:]]
    print(X.head())
    print(X.shape)



    #predictions = XGBModel(dataset.X_train, dataset.y_train, dataset.X_test)
    predictions, y_test = XGBModel(X, y)

    # Get top 4 finalists, get 4 teams with most predicted winrate wins/games_played
    #top_4_teams =

    # Evaluate the model

    evaluate(predictions, y_test)


    # Simulate 2024 bracket
    bracket_data = dataset.tournament_2024[dataset.tournament_2024['Tournament'] == 'M']
    bracket_data = bracket_data.merge(dataset.teams, on='TeamID', how='inner')
    bracket_data = bracket_data[['Seed', 'TeamName', 'TeamID']]
    #print(bracket_data.head())

    simulate_bracket(bracket_data, dataset.expand_matchup)

    #dataset.tournament_2024 = dataset.tournament_2024[]


if __name__ == '__main__':
    main()