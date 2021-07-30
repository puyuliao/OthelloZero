import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import linear
from torch.nn.modules.activation import ReLU
import torch.optim as optim

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, padding=(1,1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv3x3 = ConvBlock(in_channels, out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, stride, padding=(1,1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.conv3x3(x)
        out = self.bn(self.conv(out))
        out = self.relu(out + residual)
        return out

class Network(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_res_layers, board_size, action_size):
        super(Network, self).__init__()
        self.conv3x3 = ConvBlock(in_channels, out_channels, 1)
        res_blocks = [ResidualBlock(out_channels,out_channels, 1) for i in range(num_of_res_layers)]
        self.res_layers = nn.Sequential(*res_blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(out_channels, 2, 1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(2 * board_size , action_size)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(board_size, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
    def forward(self, x):
        out = self.conv3x3(x)
        out = self.res_layers(out)

        policy_out = self.policy_head(out)
        value_out = self.value_head(out)
        return F.log_softmax(policy_out,dim=1), torch.tanh(value_out)

if __name__ == "__main__":
    from OthelloGame import Game
    game = Game()
    nnet = Network(4, 128, 10, 64, 65)
    p,v = nnet(game.get_nnet_format())
    p = torch.exp(p)
    print(f'get p={p}, v={v}')