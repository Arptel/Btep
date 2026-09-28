import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousMCCLoss(nn.Module):
    """
    Differentiable Matthews Correlation Coefficient (MCC) Loss:
    L_MCC = 1 - MCC
    Where TP, TN, FP, FN are computed continuously without hard thresholding.
    """
    def __init__(self, eps=1e-7):
        super(ContinuousMCCLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        y_pred: (B, 1, H, W) continuous probabilities in [0, 1]
        y_true: (B, 1, H, W) ground truth binary labels in {0, 1}
        """
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        tp = torch.sum(y_pred * y_true)
        tn = torch.sum((1.0 - y_true) * (1.0 - y_pred))
        fp = torch.sum((1.0 - y_true) * y_pred)
        fn = torch.sum(y_true * (1.0 - y_pred))

        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + self.eps)

        mcc = numerator / (denominator + self.eps)
        return 1.0 - mcc


class CompoundLoss(nn.Module):
    """
    Compound Loss in SA-UNetv2 (ISBI 2026):
    L_total = lambda1 * L_BCE + lambda2 * L_MCC
    Default: lambda1 = 0.5, lambda2 = 0.5
    """
    def __init__(self, lambda_bce=0.5, lambda_mcc=0.5, eps=1e-7):
        super(CompoundLoss, self).__init__()
        self.lambda_bce = lambda_bce
        self.lambda_mcc = lambda_mcc
        self.bce = nn.BCELoss()
        self.mcc = ContinuousMCCLoss(eps=eps)

    def forward(self, y_pred, y_true):
        loss_bce = self.bce(y_pred, y_true)
        loss_mcc = self.mcc(y_pred, y_true)
        return self.lambda_bce * loss_bce + self.lambda_mcc * loss_mcc, loss_bce, loss_mcc


if __name__ == '__main__':
    criterion = CompoundLoss(0.5, 0.5)
    pred = torch.tensor([0.9, 0.8, 0.1, 0.2], requires_grad=True)
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss, bce, mcc = criterion(pred, target)
    loss.backward()
    print(f"Total Loss: {loss.item():.4f}, BCE: {bce.item():.4f}, MCC Loss: {mcc.item():.4f}")
    print(f"Pred Gradients: {pred.grad}")
