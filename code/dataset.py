
import pandas as pd
import numpy as np

from evaluate import evaluate
from model import XGBModel


DATA_PATH = "march-machine-learning-mania-2024/"

class NCAADataset():
    def __init__(self):
        # Teams
        self.teams = pd.read_csv(DATA_PATH + "MTeams.csv")

        # Regular Season Results
        season = 2023
        self.season_results = self._season_results(season)

        # Regular Season Team Stats
        self.team_stats = self._team_stats(self.season_results)


        # Create X_train, y_train
        np.random.seed(42)  # For reproducibility
        train_data = self.season_results[['WTeamID', 'LTeamID']].copy()
        train_data['Random'] = np.random.rand(len(train_data))
        train_data['Team1ID'] = np.where(train_data['Random'] < 0.5, train_data['WTeamID'], train_data['LTeamID'])
        train_data['Team2ID'] = np.where(train_data['Random'] >= 0.5, train_data['WTeamID'], train_data['LTeamID'])
        train_data['Team1Win'] = (train_data['Team1ID'] == train_data['WTeamID']).astype(int)

        # Merge team_stats to get the win rates for Team1 and Team2
        train_data = train_data.merge(self.team_stats[['TeamID', 'WinRate']], left_on='Team1ID', right_on='TeamID', how='left')
        train_data.rename(columns={'WinRate': 'Team1WinRate'}, inplace=True)
        train_data = train_data.merge(self.team_stats[['TeamID', 'WinRate']], left_on='Team2ID', right_on='TeamID', how='left')
        train_data.rename(columns={'WinRate': 'Team2WinRate'}, inplace=True)

        self.X_train = train_data[['Team1WinRate', 'Team2WinRate']]
        self.y_train = train_data['Team1Win']


        # Test Data
        self.tournament_results = self._tournament_results(season)

        # Create X_test, y_test
        test_data = self.tournament_results[['WTeamID', 'LTeamID']].copy()
        test_data = test_data.merge(self.team_stats[['TeamID', 'WinRate']], left_on='WTeamID', right_on='TeamID', how='left')
        test_data.rename(columns={'WinRate': 'Team1WinRate'}, inplace=True)
        test_data = test_data.merge(self.team_stats[['TeamID', 'WinRate']], left_on='LTeamID', right_on='TeamID', how='left')
        test_data.rename(columns={'WinRate': 'Team2WinRate'}, inplace=True)

        self.X_test = test_data[['Team1WinRate', 'Team2WinRate']]
        self.y_test = np.ones(len(test_data))


    def _tournament_results(self, season):
        tourney_results_path = DATA_PATH + "MNCAATourneyCompactResults.csv"

        tourney_results = pd.read_csv(tourney_results_path)
        tourney_results = tourney_results[tourney_results["Season"] == season]

        return tourney_results

    def _season_results(self, season):
        season_results_path = DATA_PATH + "MRegularSeasonCompactResults.csv"

        season_results = pd.read_csv(season_results_path)
        season_results = season_results[season_results["Season"] == season]

        return season_results
    
    def _team_stats(self, season_results):
        wins = season_results['WTeamID'].value_counts().reset_index()
        losses = season_results['LTeamID'].value_counts().reset_index()

        # Rename columns for clarity
        wins.columns = ['TeamID', 'Wins']
        losses.columns = ['TeamID', 'Losses']

        # Merge wins and losses DataFrames on TeamID
        team_stats = pd.merge(wins, losses, on='TeamID', how='outer').fillna(0)

        # Calculate win rate
        team_stats['WinRate'] = team_stats['Wins'] / (team_stats['Wins'] + team_stats['Losses'])

        #team_stats_sorted = team_stats.sort_values(by='WinRate', ascending=False)

        #top_teams = pd.merge(team_stats_sorted, teams, left_on='TeamID', right_on='TeamID')

        #top_5_teams = top_teams.head(5)

        # Display the names and win rates of the top 5 teams
        #print(top_5_teams[['TeamName', 'WinRate']])

        return team_stats


if __name__ == '__main__':

    dataset = NCAADataset()
    season = 2023
    
    print("Teams")
    print(dataset.teams.head())

    print("Season Results")
    print(dataset.season_results.head())
    print(dataset.season_results.shape)


    print("Team Stats")
    print(dataset.team_stats.head())


    print("Tournament Results")
    print(dataset.tournament_results.head())
    print(dataset.tournament_results.shape)

    """
    # Get the last games
    tourney_results_rev = tourney_results.sort_values('DayNum', ascending=False)
    last_games = tourney_results_rev[:3]

    # Get teams in last_games
    top_4_teams = last_games['WTeamID'].tolist() + last_games['LTeamID'].tolist()

    # Get the team names
    top_4_teams = dataset.teams[dataset.teams['TeamID'].isin(top_4_teams)]

    # Display the top 4 teams
    print("Top 4 Teams")
    print(top_4_teams['TeamName'])
    """


    # Prepare features and target
    #features = tourney_results[['Team1WinRate', 'Team2WinRate']]
    #target = tourney_results['Team1Win']

    