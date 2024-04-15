import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class NNPredictor(nn.Module):
    def __init__(self, data):
        self.train_loader, self.test_loader = self._load_and_prepare_data(data)

        ### Neural Network
        super(NNPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.train_loader.dataset.tensors[0].shape[1], 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def _load_and_prepare_data(self, data):
        y = (data['T1_Score'] > data['T2_Score']).astype(int)
        X = data[data.columns[6:]]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

        # Normalize the features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # Datasets and Loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        return DataLoader(train_dataset, batch_size=32, shuffle=True), DataLoader(test_dataset, batch_size=32, shuffle=False)


    def forward(self, x):
        return self.model(x)


    def train_model(self):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training
        self.model.train()
        for epoch in range(50):
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        # Evaluation
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')

        


    def predict_matchup(self, matchup):
        """
        Predict outcome of a game between two teams.

        Parameters:
        matchup

        Returns:
        bool: Predicts Team1 wins the game.
        """
        
        # Extract features and apply the same preprocessing as training data
        features = self.scaler.transform(matchup.values)  # Apply scaling

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Model prediction
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self(features_tensor)
            predicted_probability = output.item()  # get the scalar value of the tensor

        print(predicted_probability)
        return predicted_probability >= 0.5