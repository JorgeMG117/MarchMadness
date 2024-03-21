
import pandas as pd
import numpy as np

from evaluate import evaluate
from model import XGBModel


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

    #print(teams.head())

    #print(season_results.head())
    #print(season_results.shape)

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
    #print(top_5_teams[['TeamName', 'WinRate']])


    # MNCAATourneyCompactResults
    # This are the games to be predicted
    # Maybe try using the kaggle format of the test file
    tourney_results_path = DATA_PATH + "MNCAATourneyCompactResults.csv"
    tourney_results = pd.read_csv(tourney_results_path)
    tourney_results = tourney_results[tourney_results["Season"] == season]
    #print(tourney_results.head())
    #print(tourney_results.shape)
    # Get the last games
    tourney_results_rev = tourney_results.sort_values('DayNum', ascending=False)
    last_games = tourney_results_rev[:3]

    # Get teams in last_games
    top_4_teams = last_games['WTeamID'].tolist() + last_games['LTeamID'].tolist()

    # Get the team names
    top_4_teams = teams[teams['TeamID'].isin(top_4_teams)]

    # Display the top 4 teams
    print("Top 4 Teams")
    print(top_4_teams['TeamName'])


    # Randomly assign Team1 and Team2
    np.random.seed(42)  # For reproducibility
    tourney_results['Random'] = np.random.rand(len(tourney_results))
    tourney_results['Team1ID'] = np.where(tourney_results['Random'] < 0.5, tourney_results['WTeamID'], tourney_results['LTeamID'])
    tourney_results['Team2ID'] = np.where(tourney_results['Random'] >= 0.5, tourney_results['WTeamID'], tourney_results['LTeamID'])
    tourney_results['Team1Win'] = (tourney_results['Team1ID'] == tourney_results['WTeamID']).astype(int)

    # Merge team_stats to get the win rates for Team1 and Team2
    tourney_results = tourney_results.merge(team_stats[['TeamID', 'WinRate']], left_on='Team1ID', right_on='TeamID', how='left')
    tourney_results.rename(columns={'WinRate': 'Team1WinRate'}, inplace=True)
    tourney_results = tourney_results.merge(team_stats[['TeamID', 'WinRate']], left_on='Team2ID', right_on='TeamID', how='left')
    tourney_results.rename(columns={'WinRate': 'Team2WinRate'}, inplace=True)

    # Prepare features and target
    features = tourney_results[['Team1WinRate', 'Team2WinRate']]
    target = tourney_results['Team1Win']

    print(tourney_results.shape)
    print(target.shape)


    predictions, y_test = XGBModel(features, target)


    # Get top 4 finalists, get 4 teams with most predicted winrate wins/games_played
    #top_4_teams =

    # Evaluate the model
    evaluate(predictions, y_test)