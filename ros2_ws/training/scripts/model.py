import torch
import torch.nn as nn


class F1CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )

        with torch.no_grad():

            dummy = torch.zeros(1, 3, 80, 200)

            features = self.features(dummy)

            flattened_size = features.numel()

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(flattened_size, 100),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(100, 50),
            nn.ReLU(),

            nn.Linear(50, 10),
            nn.ReLU(),

            nn.Linear(10, 2)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.regressor(x)

        return x
