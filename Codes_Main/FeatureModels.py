import torch
import torch.nn as nn
from torch.autograd import Variable
from Models import CCC, SincModel

class FeatureModel0(nn.Module):
    def __init__(self, featSize, device="cuda"):
        super(FeatureModel0, self).__init__()
        self.featSize = featSize
        self.device = device
        # self.out = nn.Linear(self.featSize, 1)
        self.lin = nn.Sequential(
            nn.Linear(featSize, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        batch_size, timesteps, sq_len = x.size()
        output = self.lin(x)
        output = output.view(batch_size, 1, timesteps, 1)
        return output

class FeatureModel1(nn.Module):
    def __init__(self, featSize, device="cuda", kernel=(64,1), channels=16):
        super(FeatureModel1, self).__init__()
        self.featSize = featSize
        self.device = device
        self.conv1=nn.Conv2d(in_channels=1,out_channels=channels, kernel_size=kernel,stride=1, padding=0, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.deconv1=nn.ConvTranspose2d(in_channels=channels,out_channels=1,kernel_size=kernel, bias=False)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.out = nn.Linear(self.featSize, 1)

    def forward(self, x):
        batch_size, timesteps, sq_len = x.size()
        x = x.view(batch_size, 1, timesteps, sq_len)
        output = self.conv1(x)
        output = self.deconv1(output)
        output = self.out(output)
        return output


class FeatureModel2(nn.Module):
    def __init__(self, featSize, device="cuda"):
        super(FeatureModel2, self).__init__()
        self.featSize = featSize
        self.device = device
        self.rnn = nn.GRU(
            input_size=featSize, 
            hidden_size=128, 
            num_layers=2,
            batch_first=True)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        batch_size, timesteps, sq_len = x.size()
        x = x.view(batch_size, timesteps, sq_len)
        output, _ = self.rnn(x)
        output = self.out(output)
        output = output.view(batch_size, 1, timesteps, 1)
        return output


class FeatureModel3(nn.Module):
    def __init__(self, featSize, device="cuda"):
        super(FeatureModel3, self).__init__()
        self.featSize = featSize
        self.device = device
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3, 1),stride=1, padding=0, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=(4, 1),stride=1, padding=0, bias=False)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128, kernel_size=(5, 1),stride=1, padding=0, bias=False)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.deconv1=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(5, 1), bias=False)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.deconv2=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(4, 1), bias=False)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.deconv3=nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=(3, 1), bias=False)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.out = nn.Linear(self.featSize, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size, timesteps, sq_len = x.size()
        x = x.view(batch_size, 1, timesteps, sq_len)
        output = self.conv1(x)
        output = self.tanh(output)
        output = self.conv2(output)
        output = self.tanh(output)
        output = self.conv3(output)
        output = self.tanh(output)
        output = self.deconv1(output)
        output = self.tanh(output)
        output = self.deconv2(output)
        output = self.tanh(output)
        output = self.deconv3(output)
        output = self.tanh(output)
        output = self.out(output)
        return output
