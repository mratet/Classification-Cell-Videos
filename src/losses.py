import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


class WeightedCrossEntropy(nn.Module):

    def __init__(self, alpha, eps=1e-10, device="cuda"):
        super(WeightedCrossEntropy, self).__init__()

        self.alpha = alpha
        self.eps = eps

        weight_matrix = np.array(
            [
                [0, 1, 6, 10, 10, 10, 10, 10],
                [1, 0, 3, 10, 10, 10, 10, 10],
                [6, 3, 0, 2, 9, 10, 10, 10],
                [10, 10, 2, 0, 9, 9, 10, 10],
                [10, 10, 9, 9, 0, 8, 8, 8],
                [10, 10, 10, 9, 8, 0, 9, 8],
                [10, 10, 10, 10, 8, 9, 0, 9],
                [10, 10, 10, 10, 8, 8, 9, 0],
            ]
        )

        weight_matrix = weight_matrix /np.max(weight_matrix)
        self.weight_matrix = torch.tensor(weight_matrix).float().to(device=device)

    def forward(self, input, target):
        """
        Loss utilisé lors de l'entrainement des réseaux de neurones

        Input : y_pred, Vecteur (batch_size, 8) : prédiction du NN
        Attention Softmax ici donc pas besoin de l'inclure à la fin du réseau !

        Target : y_true, Vecteur (batch_size, 1) : label encodé incrémentalement
        """
        n_classes = 8

        # Compute the softmax/log-softmax of the predictions
        probs = F.softmax(input, dim=1)
        log_probs = F.log_softmax(input + self.eps, dim=1)

        # Create a one-hot encoding of the true labels
        target = F.one_hot(target, num_classes=n_classes).float()

        CrossEntropy = - torch.mean(torch.sum(target * log_probs, dim=1))

        WeightedCE = -torch.mean(torch.sum(torch.bmm(torch.unsqueeze(torch.matmul(target, self.weight_matrix), 1), torch.unsqueeze(torch.log(1 - probs + self.eps), 2)), dim=1))
        # torch.sum --> Calcul de l'erreur par sample
        # torch.mean --> on prend la moyenne sur l'ensemble des échantillons

        loss = self.alpha * CrossEntropy + (1-self.alpha) * WeightedCE

        return loss


class WeightedCrossError(nn.Module):
    def __init__(self, device="cuda"):
        super(WeightedCrossError, self).__init__()

        self.device = device
        weight_matrix = np.array(
            [
                [0, 1, 6, 10, 10, 10, 10, 10],
                [1, 0, 3, 10, 10, 10, 10, 10],
                [6, 3, 0, 2, 9, 10, 10, 10],
                [10, 10, 2, 0, 9, 9, 10, 10],
                [10, 10, 9, 9, 0, 8, 8, 8],
                [10, 10, 10, 9, 8, 0, 9, 8],
                [10, 10, 10, 10, 8, 9, 0, 9],
                [10, 10, 10, 10, 8, 8, 9, 0],
            ]
        )
        self.weight_matrix = weight_matrix / np.max(weight_matrix)

    def forward(self, input, target):
        n_classes = 8
        n = len(target)

        probs = F.softmax(input, dim=1)
        y_pred = probs.cpu().detach().numpy()
        y_pred = np.argmax(y_pred, axis=1)

        y_true = target.cpu().detach().numpy()

        conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

        error = np.multiply(conf_mat, self.weight_matrix).sum() / n

        return error

