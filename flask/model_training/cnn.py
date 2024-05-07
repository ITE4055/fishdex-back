import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CNN, self).__init__()
        self.out_dim = out_dim
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, 8, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=1, padding=0),
            nn.ReLU()
        )
        
        self.q1 = nn.Linear(22*22*32, 64) # The 22*22 are the final dimensions of the "image"
        self.q2 = nn.Linear(64, out_dim)
    
    def forward(self, x):
        conv = self.convs(x)
        flat = conv.reshape(-1, 22*22*32)
        q1 = F.relu(self.q1(flat))
        q = self.q2(q1)
        return q