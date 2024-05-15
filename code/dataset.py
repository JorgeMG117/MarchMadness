import pandas as pd

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

        # Add last 14 days
        #matchup = pd.merge(matchup, self.last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
        #matchup = pd.merge(matchup, self.last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')

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


        
        # Add last 14 days
        """
        #print("Adding last 14 days stats")
        last14days_stats_T1 = season_results.loc[season_results.DayNum>118].reset_index(drop=True)
        last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff']>0,1,0)
        last14days_stats_T1 = last14days_stats_T1.groupby(['Season','T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

        last14days_stats_T2 = season_results.loc[season_results.DayNum>118].reset_index(drop=True)
        last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff']<0,1,0)
        last14days_stats_T2 = last14days_stats_T2.groupby(['Season','T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')

        tournament_results = pd.merge(tournament_results, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
        tournament_results = pd.merge(tournament_results, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')
        self.last14days_stats_T1 = last14days_stats_T1
        self.last14days_stats_T2 = last14days_stats_T2
        """
        

        # Add team quality

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



if __name__ == '__main__':

    dataset = NCAADataset()
    
    print("### Teams")
    print(dataset.teams.head())

    print("### Season Results")
    print(dataset.season_results.head())
    print(dataset.season_results.shape)


    #print("Team Stats")
    #print(dataset.team_stats.head())


    #print("Tournament Results")
    #print(dataset.tournament_results.head())
    #print(dataset.tournament_results.shape)

    print("### Tournament Data")
    print(dataset.tournament_data.head())
    print(dataset.tournament_data.tail())
    print(dataset.tournament_data.columns)
    print(dataset.tournament_data.shape)
