
from dataset import NCAADataset


if __name__ == '__main__':
    dataset = NCAADataset()
    
    games = dataset.tournament_2024

    teams = dataset.teams

    # I want to get a df with teamID, teamName
    results = games.merge(teams, left_on='TeamID', right_on='TeamID')
    results = results[['TeamID', 'TeamName']]

    # Create file with content
    results.to_csv('predictions/results.csv', index=False)

    # Edit file to add how many games every team has won
