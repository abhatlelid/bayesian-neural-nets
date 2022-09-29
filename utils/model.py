
from torch import nn

class Net(nn.Module):
    """The network."""
    def __init__(self, in_feat=3, out_feat=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=out_feat)
        )

    def forward(self, x):
        x = self.layers(x)
        return x