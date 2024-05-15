from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class NNPredictor(nn.Module):
    def __init__(self, data):
        self.scaler = StandardScaler()
        self.batch_size = 32
        self.epochs = 20

        self.train_loader, self.test_loader = self._load_and_prepare_data(data)

        print(self.train_loader.dataset.tensors[0].shape[1])

        ### Neural Network
        super(NNPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.train_loader.dataset.tensors[0].shape[1], 32),#64, 512
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def _load_and_prepare_data(self, data):
        y = (data['T1_Score'] > data['T2_Score']).astype(int)
        X = data[data.columns[6:]]


        #print("Target: ")
        #print(y.head())
        #print("Shape of y: " + str(y.shape))

        #print("Features: ")
        #print(X.head())
        #print("Shape of X: " + str(X.shape))

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

        # Normalize the features
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

        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


    def forward(self, x):
        return self.model(x)


    def train_model(self):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.03)

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        train_precisions = []
        train_recalls = []
        test_precisions = []
        test_recalls = []


        # Early stopping
        best_val_loss = float('inf')
        early_stop_counter = 0
        patience = 3

        # Learning Rate Scheduling
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Training
        for epoch in range(self.epochs):
            self.model.train()
            train_loss, correct, total, tp, fp, fn = 0, 0, 0, 0, 0, 0

            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Sum up batch loss
                train_loss += loss.item() * inputs.size(0)
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            train_losses.append(train_loss / len(self.train_loader.dataset))
            train_accuracies.append(100 * correct / total)
            train_precisions.append(precision)
            train_recalls.append(recall)

            # Evaluation
            self.model.eval()
            test_loss, correct, total, tp, fp, fn = 0, 0, 0, 0, 0, 0
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    tp += ((predicted == 1) & (labels == 1)).sum().item()
                    fp += ((predicted == 1) & (labels == 0)).sum().item()
                    fn += ((predicted == 0) & (labels == 1)).sum().item()

            # Calculate metrics for test
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_losses.append(test_loss / len(self.test_loader.dataset))
            test_accuracies.append(100 * correct / total)
            test_precisions.append(precision)
            test_recalls.append(recall)
            # TODO: Que hacer con precision y recall??

            # Early stopping
            if test_losses[-1] < best_val_loss:
                best_val_loss = test_losses[-1]
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break


            # Learning Rate Scheduling
            scheduler.step()


            #if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]}, Train Accuracy: {train_accuracies[-1]}%, '
                f'Val Loss: {test_losses[-1]}, Val Accuracy: {test_accuracies[-1]}%')


        self._plot_training(train_losses, test_losses, train_accuracies, test_accuracies)
        
    def _plot_training(self, train_losses, test_losses, train_accuracies, test_accuracies):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(test_accuracies, label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.savefig('img/train_val_loss_acc.png')
        plt.show()


    def predict_matchup(self, matchup):
        """
        Predict outcome of a game between two teams.

        Parameters:
        matchup

        Returns:
        bool: Predicts Team1 wins the game.
        """
        #print(matchup.shape)
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
    
    def __str__(self):
        # Initialize an empty string to store the network structure
        network_str = ""
        
        # Iterate through the layers of the model
        for i, layer in enumerate(self.model):
            # Add the layer type and its parameters to the string
            network_str += f"Layer {i+1}: {layer.__class__.__name__}"
            # Add the number of input and output features if applicable
            if hasattr(layer, "in_features"):
                network_str += f" (Input: {layer.in_features}, Output: {layer.out_features})"
            network_str += "\n"
        
        return network_str