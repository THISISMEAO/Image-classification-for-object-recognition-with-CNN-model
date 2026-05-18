'''
Concrete Evaluate class for multiple evaluation metrics
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')

        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        #Convert tensors to numpy arrays if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        #Classification metrics
        acc = accuracy_score(y_true, y_pred)

        precision_weighted = precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        precision_macro = precision_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_micro = precision_score(
            y_true, y_pred, average='micro', zero_division=0
        )

        recall_weighted = recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        recall_macro = recall_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        recall_micro = recall_score(
            y_true, y_pred, average='micro', zero_division=0
        )

        f1_weighted = f1_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        f1_macro = f1_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        f1_micro = f1_score(
            y_true, y_pred, average='micro', zero_division=0
        )

        #AUC calculation
        auc_macro = None

        if 'y_score' in self.data and self.data['y_score'] is not None:
            y_score = self.data['y_score']

            if isinstance(y_score, torch.Tensor):
                y_score = y_score.cpu().numpy()

            #Number of classes comes from model output columns
            num_classes = y_score.shape[1]
            classes = np.arange(num_classes)

            y_true_bin = label_binarize(y_true, classes=classes)

            try:
                auc_macro = roc_auc_score(
                    y_true_bin,
                    y_score,
                    average='macro',
                    multi_class='ovr'
                )
            except ValueError:
                auc_macro = None

        #Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        #Print metrics
        print("Accuracy:", acc)
        print("Precision (weighted):", precision_weighted)
        print("Precision (macro):", precision_macro)
        print("Precision (micro):", precision_micro)
        print("Recall (weighted):", recall_weighted)
        print("Recall (macro):", recall_macro)
        print("Recall (micro):", recall_micro)
        print("F1 (weighted):", f1_weighted)
        print("F1 (macro):", f1_macro)
        print("F1 (micro):", f1_micro)

        if auc_macro is not None:
            print("AUC (macro, OvR):", auc_macro)

        print("Confusion Matrix:")
        print(cm)

        return {
            'acc': acc,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'recall_weighted': recall_weighted,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'precision_weighted': precision_weighted,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'auc_macro': auc_macro,
            'confusion_matrix': cm
        }