from dataset import NCAADataset
from model import XGBModel
from evaluate import evaluate
import numpy as np

def simulate_bracket(tournament_seeds):
    pass


def main():
    # Create dataset
    dataset = NCAADataset()

    print(dataset.tournament_data.head())
    print(dataset.tournament_data.shape)
    print(dataset.tournament_data.columns)

    y = dataset.tournament_data['T1_Score'] - dataset.tournament_data['T2_Score']
    y = np.where(y > 0, 1, 0)
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
    #dataset.tournament_2024
    simulate_bracket(dataset.tournament_2024)

    #dataset.tournament_2024 = dataset.tournament_2024[]


if __name__ == '__main__':
    main()