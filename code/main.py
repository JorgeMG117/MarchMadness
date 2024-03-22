from dataset import NCAADataset
from model import XGBModel
from evaluate import evaluate


def main():
    # Create dataset
    dataset = NCAADataset()


    predictions = XGBModel(dataset.X_train, dataset.y_train, dataset.X_test)

    # Get top 4 finalists, get 4 teams with most predicted winrate wins/games_played
    #top_4_teams =

    # Evaluate the model
    evaluate(predictions, dataset.y_test)

if __name__ == '__main__':
    main()