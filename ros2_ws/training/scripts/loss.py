import torch.nn as nn


class WeightedControlLoss(nn.Module):

    def __init__(self, v_weight=0.4, w_weight=0.6):

        super().__init__()

        self.v_weight = v_weight
        self.w_weight = w_weight

        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):

        loss_v = self.mse(
            predictions[:, 0],
            targets[:, 0]
        )

        loss_w = self.mse(
            predictions[:, 1],
            targets[:, 1]
        )

        total_loss = (
            self.v_weight * loss_v
            + self.w_weight * loss_w
        )

        return total_loss
