import torch
import torch.nn as nn

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature
        self.crit = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        B = z1.shape[0]
        reps = torch.cat([z1, z2], dim=0)
        sim = torch.mm(reps, reps.t()) / self.temp
        # Mask self-similarity
        sim.masked_fill_(torch.eye(2*B, device=z1.device).bool(), -1e9)
        # Labels: i -> i+B, i+B -> i
        labels = torch.cat([torch.arange(B)+B, torch.arange(B)], dim=0).to(z1.device).long()
        return self.crit(sim, labels)