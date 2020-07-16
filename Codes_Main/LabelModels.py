import torch
import torch.nn as nn
from torch.autograd import Variable
from Models import CCC, SincModel

class LabelModel1(nn.Module):
    def __init__(self, featSize, device="cuda", pretaus=None):
        super(LabelModel1, self).__init__()
        self.featSize = featSize
        self.device = device

    def forward(self, x):
        batch_size, timesteps, sq_len = x.size()
        output = x.view(batch_size, 1, timesteps, sq_len)
        return output


class LabelModel2(nn.Module):
    def __init__(self, featSize, fc=0.25, fs=25, M=10, device="cuda", pretaus=None):
        super(LabelModel2, self).__init__()
        self.featSize = featSize
        self.device = device
        self.sinc = SincModel(featSize, fc=fc, fs=fs, M=M, device=device, pretaus=pretaus)

    def forward(self, x):
        output = self.sinc(x)
        # batch_size, timesteps, sq_len = x.size()
        # output = x.view(batch_size, 1, timesteps, sq_len)
        return output


class LabelModel3(nn.Module):
    def __init__(self, featSize, fc=0.25, fs=25, M=10, device="cuda", pretaus=None, preweights=None):
        super(LabelModel3, self).__init__()
        self.featSize = featSize
        self.device = device
        self.sinc = SincModel(featSize, fc=fc, fs=fs, M=M, device=device, pretaus=pretaus)
        self.weights = preweights if (not preweights is None) else torch.nn.Parameter(torch.rand(featSize, device=self.device, requires_grad=True).float())
        self.softmax = nn.Softmax(dim=0)
        self.allSinc = SincModel(1, fc=fc, fs=fs, M=M, device=device, pretaus=None)

    def forward(self, x):
        batch_size, timesteps, sq_len = x.size()
        output = self.sinc(x)
        weights = self.softmax(self.weights)
        output = output * weights
        output = torch.sum(output, 3).unsqueeze(3)
        output = self.allSinc(output.view(batch_size, timesteps, 1))
        # output = output.view(batch_size, 1, timesteps, 1)
        return output
