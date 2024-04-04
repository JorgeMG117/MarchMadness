
import pandas as pd
import numpy as np


DATA_PATH = "march-machine-learning-mania-2024/"

class NCAADataset():
    def __init__(self):
        # Teams
        self.teams = pd.read_csv(DATA_PATH + "MTeams.csv")

        # Seeds
        self.seeds = pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv")
        #print(self.seeds.head())

        # Regular Season Data
        self.season_results = self._prepare_data(DATA_PATH + "MRegularSeasonDetailedResults.csv")
        #print(self.season_results.columns)
        #print(self.season_results.head())
        #print(self.season_results.shape)

        # Tournament Data
        self.tournament_results = self._prepare_data(DATA_PATH + "MNCAATourneyDetailedResults.csv")
        #print(self.tournament_results.head())
        #print(self.tournament_results.shape)


        # Feature Engineering
        self.tournament_data = self._feature_engineering(self.season_results, self.tournament_results, self.seeds)

        # 2024 Tournament Games
        self.tournament_2024 = pd.read_csv(DATA_PATH + "2024_tourney_seeds.csv")


    def expand_matchup(self, matchup):
        matchup = pd.merge(matchup, self.season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
        matchup = pd.merge(matchup, self.season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')

        matchup = pd.merge(matchup, self.seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
        matchup = pd.merge(matchup, self.seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

        matchup["Seed_diff"] = matchup["T1_seed"] - matchup["T2_seed"]

        return matchup
        

    def _prepare_data(self, path):
        season_results = pd.read_csv(path)
        
        # Swap contains the same data as season_results, but with the winning and losing teams swapped
        season_results_swap = season_results.copy()

        season_results_swap.loc[season_results['WLoc'] == 'H', 'WLoc'] = 'A'
        season_results_swap.loc[season_results['WLoc'] == 'A', 'WLoc'] = 'H'
        season_results.columns.values[6] = 'location'
        season_results_swap.columns.values[6] = 'location'

        season_results.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(season_results.columns)]
        season_results_swap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(season_results_swap.columns)]

        # Order columns to follow same structure
        season_results_swap = season_results_swap[season_results.columns]


        # Mix them together
        season_results = pd.concat([season_results, season_results_swap]).sort_index().reset_index(drop = True)

        season_results.loc[season_results.location=='N','location'] = '0'
        season_results.loc[season_results.location=='H','location'] = '1'
        season_results.loc[season_results.location=='A','location'] = '-1'
        season_results.location = season_results.location.astype(int)

        season_results['PointDiff'] = season_results['T1_Score'] - season_results['T2_Score']

        #print(season_results.head())

        #season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(np.mean)
        #season_statistics.head()

        return season_results
    

    def _feature_engineering(self, season_results, tournament_results, seeds):
        # Group each team stats by season
        team_prefixes = ['T1_', 'T2_']
        metrics = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
        boxscore_cols = [f"{prefix}{metric}" for prefix in team_prefixes for metric in metrics] + ['PointDiff']

        season_statistics = season_results.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg('mean').reset_index()


        # Duplicate season_statistics for T1 and T2
        season_statistics_T1 = season_statistics.copy()
        season_statistics_T2 = season_statistics.copy()

        season_statistics_T1.columns = ["T1_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T1.columns)]
        season_statistics_T2.columns = ["T2_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T2.columns)]
        season_statistics_T1.columns.values[0] = "Season"
        season_statistics_T2.columns.values[0] = "Season"


        # Add season statistics to tournament games
        tournament_results = tournament_results[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID' ,'T2_Score']]# TODO POdemos utilizar T2Score y T1Score????

        tournament_results = pd.merge(tournament_results, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
        tournament_results = pd.merge(tournament_results, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')


        # Add seed difference
        seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))

        seeds_T1 = seeds[['Season','TeamID','seed']].copy()
        seeds_T2 = seeds[['Season','TeamID','seed']].copy()
        seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
        seeds_T2.columns = ['Season','T2_TeamID','T2_seed']

        tournament_results = pd.merge(tournament_results, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
        tournament_results = pd.merge(tournament_results, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

        tournament_results["Seed_diff"] = tournament_results["T1_seed"] - tournament_results["T2_seed"]


        # To use it for the 2024 playoffs
        self.seeds_T1 = seeds_T1
        self.seeds_T2 = seeds_T2
        self.season_statistics_T1 = season_statistics_T1
        self.season_statistics_T2 = season_statistics_T2

        return tournament_results


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
    
    #print("Teams")
    #print(dataset.teams.head())

    #print("Season Results")
    #print(dataset.season_results.head())
    #print(dataset.season_results.shape)


    #print("Team Stats")
    #print(dataset.team_stats.head())


    #print("Tournament Results")
    #print(dataset.tournament_results.head())
    #print(dataset.tournament_results.shape)

    print("Tournament Data")
    print(dataset.tournament_data.head())
    print(dataset.tournament_data.shape)

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

    