
import pandas as pd
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_PATH = "march-machine-learning-mania-2024/"

class NCAADataset():
    def __init__(self):
        # Teams (pandas dataframe)

        # self.features

        pass


if __name__ == '__main__':

    teams_path = DATA_PATH + "MTeams.csv"

    teams = pd.read_csv(teams_path)

    season_results_path = DATA_PATH + "MRegularSeasonCompactResults.csv"

    season = 2023
    season_results = pd.read_csv(season_results_path)
    season_results = season_results[season_results["Season"] == season]

    print(teams.head())

    print(season_results.head())

    wins = season_results['WTeamID'].value_counts().reset_index()
    losses = season_results['LTeamID'].value_counts().reset_index()

    # Rename columns for clarity
    wins.columns = ['TeamID', 'Wins']
    losses.columns = ['TeamID', 'Losses']

    # Merge wins and losses DataFrames on TeamID
    team_stats = pd.merge(wins, losses, on='TeamID', how='outer').fillna(0)

    # Calculate win rate
    team_stats['WinRate'] = team_stats['Wins'] / (team_stats['Wins'] + team_stats['Losses'])

    team_stats_sorted = team_stats.sort_values(by='WinRate', ascending=False)

    top_teams = pd.merge(team_stats_sorted, teams, left_on='TeamID', right_on='TeamID')

    top_5_teams = top_teams.head(5)

    # Display the names and win rates of the top 5 teams
    print(top_5_teams[['TeamName', 'WinRate']])


    
    """
    # Assuming 'team_stats' is your DataFrame containing TeamID and WinRate
    # Assuming 'matches' is your DataFrame containing the matchups you want to predict, 
    # with 'Team1ID' and 'Team2ID' columns for the IDs of the teams in each game
    # and 'Team1Win' column where 1 indicates Team1 wins, and 0 indicates Team2 wins

    # Merge team_stats to get the win rates for Team1 and Team2 in each game
    matches = matches.merge(team_stats[['TeamID', 'WinRate']], left_on='Team1ID', right_on='TeamID')
    matches.rename(columns={'WinRate': 'Team1WinRate'}, inplace=True)
    matches = matches.merge(team_stats[['TeamID', 'WinRate']], left_on='Team2ID', right_on='TeamID')
    matches.rename(columns={'WinRate': 'Team2WinRate'}, inplace=True)

    # Prepare features and target
    features = matches[['Team1WinRate', 'Team2WinRate']]
    target = matches['Team1Win']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

"""


    # MNCAATourneyCompactResults
    # This are the games to be predicted
    # Maybe try using the kaggle format of the test file
    tourney_results_path = DATA_PATH + "MNCAATourneyCompactResults.csv"
    tourney_results = pd.read_csv(tourney_results_path)
    tourney_results = tourney_results[tourney_results["Season"] == season]
    print(tourney_results.head())
    print(tourney_results.shape)