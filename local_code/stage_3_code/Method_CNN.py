import os
from local_code.base_class import dataset
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 50 #change if necessary
    learning_rate = 1e-2 #change if necessary

    def __init__(
        self,
        mName,
        mDescription,
        input_channels=3,
        num_classes=10,
        input_height=32,
        input_width=32,
        learning_rate=0.001,
        max_epoch=20,
        batch_size=64,
        dropout_rate=0.5,
        optimizer_name='adam',
        dataset_name='CNN',
    ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.epoch_numbers = []
        self.train_losses = []
        self.train_accuracies = []

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.dropout = nn.Dropout(dropout_rate)
        self.dataset_name = dataset_name

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2, stride=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        conv_output_height = input_height // 4
        conv_output_width = input_width // 4

        self.fc1 = nn.Linear(64 * conv_output_height * conv_output_width, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        #already put in datatset loader
         # x shape coming in: (batch, 3, H, W) — take just 1 channel
        #x = x[:, 0:1, :, :] # → (batch, 1, H, W)

        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # → (batch, 32, H/2, W/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # → (batch, 64, H/4, W/4)

        x = torch.flatten(x, 1) # → (batch, 64*H/4*W/4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # → (batch, num_classes)
        return x

    def train_model(self, X, y):  #Name changed due to error in switching to eval mode
        
        if self.optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9
            )
        elif self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        dataset    = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        epoch_numbers = []  #Data collection during training added for loss plots
        train_losses = []
        train_accuracies = []

        for epoch in range(self.max_epoch):
            self.train()  # set to train mode each epoch
            epoch_loss = 0

            for X_batch, y_batch in dataloader:
                y_pred = self.forward(X_batch)
                loss   = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # evaluate on full set each epoch
            self.eval()
            with torch.no_grad():
                y_pred_full = self.forward(X_tensor)

            accuracy_evaluator.data = {
                'true_y': y_tensor,
                'pred_y': y_pred_full.max(1)[1]
            }
            avg_loss = epoch_loss / len(dataloader)
            acc = accuracy_evaluator.evaluate()['acc']

            print(f'Epoch: {epoch}  Accuracy: {acc:.4f}  Loss: {avg_loss:.4f}')
            epoch_numbers.append(epoch + 1)
            train_losses.append(avg_loss)
            train_accuracies.append(acc)

        return epoch_numbers, train_losses, train_accuracies

    def test(self, X):
        self.eval()  # switch to evaluation mode
        X_tensor = torch.FloatTensor(np.array(X))
    
        with torch.no_grad():  #no gradients needed as during testing, we don’t update weights
            y_pred = self.forward(X_tensor)
            y_probs  = torch.softmax(y_pred, dim=1)

        return y_pred.max(1)[1], y_probs
    
    def plot_metrics(self, epoch_numbers, train_losses, train_accuracies, y_true, y_probs):

        os.makedirs('./result/stage_3_result/', exist_ok=True)

        # -----------------------------
        # Plot training loss
        # -----------------------------
        plt.figure()
        plt.plot(epoch_numbers, train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('CNN Training Loss')
        plt.savefig(f'./result/stage_3_result/{self.dataset_name}_training_loss.png')
        plt.close()

        # -----------------------------
        # Plot training accuracy
        # -----------------------------
        plt.figure()
        plt.plot(epoch_numbers, train_accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.title('CNN Training Accuracy')
        plt.savefig(f'./result/stage_3_result/{self.dataset_name}_training_accuracy.png')
        plt.close()

        # -----------------------------
        # Plot ROC curves
        # -----------------------------
        y_true = np.array(y_true).astype(int)
        y_probs = np.array(y_probs)

        num_classes = y_probs.shape[1]
        classes = list(range(num_classes))

        print("ROC y_true unique:", np.unique(y_true))
        print("ROC y_probs shape:", y_probs.shape)

        y_true_bin = label_binarize(y_true, classes=classes)

        plt.figure()

        for i in range(num_classes):
            # Skip class if it does not appear in this debug/test split
            if y_true_bin[:, i].sum() == 0:
                continue

            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} AUC = {roc_auc:.2f}')

        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('CNN ROC Curve')
        plt.legend()
        plt.savefig(f'./result/stage_3_result/{self.dataset_name}_ROC_curve.png')
        plt.close()

    def run(self):
        print('method running...')
        print('--start training...')
        epoch_numbers, train_losses, train_accuracies = self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y, y_probs = self.test(self.data['test']['X'])

        self.epoch_numbers = epoch_numbers
        self.train_losses = train_losses
        self.train_accuracies = train_accuracies
        self.y_probs = y_probs.numpy()

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'y_score': y_probs
        }