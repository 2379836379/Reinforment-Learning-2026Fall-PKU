import torch
from torch import nn

class QNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, lr, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)
    
    def inference(self, obs):
        q_value = self.seq(obs)
        return q_value
    
    def optimize(self, loss, max_grad_norm=0.0):
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = None
        if max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm).item()
        self.optimizer.step()
        return grad_norm


class BreakoutQNetwork(QNetwork):
    """DQN CNN for uint8 observations shaped (4, 80, 80) or (N, 4, 80, 80)."""

    def __init__(self, output_dim, lr, hidden_dim=512, dueling=False):
        nn.Module.__init__(self)
        self.dueling = dueling
        self.seq = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        if dueling:
            # Reuse the convolutional encoder; each head gets its own hidden layer.
            self.seq = self.seq[:7]
            self.value = nn.Sequential(nn.Linear(64 * 6 * 6, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            self.advantage = nn.Sequential(nn.Linear(64 * 6 * 6, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def inference(self, obs):
        single = obs.ndim == 3
        if single:
            obs = obs.unsqueeze(0)
        q_values = self.seq(obs.float() / 255.0)
        if self.dueling:
            advantage = self.advantage(q_values)
            q_values = self.value(q_values) + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values.squeeze(0) if single else q_values
