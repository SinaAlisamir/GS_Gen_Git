import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class SincModel(nn.Module):
    '''
        fc:  the cutoff frequency in Hz.
        fs:  the sampling frequency in Hz.
        M:   half of the signal in seconds. Total number of points is N = 2*M*fs + 1.
        tau: the time shift for the sinc kernel.
    '''
    def __init__(self, featSize, fc=5, fs=25, M=10, device="cuda", pretaus=None):
        super(SincModel, self).__init__()
        self.featSize = featSize
        self.device = device
        # self.kernel_size = kernel_size
        self.taus = pretaus if (not pretaus is None) else torch.nn.Parameter(torch.zeros(featSize, device=self.device).float())
        # print(self.taus.size())
        self.M = M
        self.fs = fs
        self.shift = self.M*self.fs
        self.fc = fc
        self.kernel_size = 2*M*fs + 1

    def MakeSincKernel(self, fc, fs, M, tau):
        N = 2*M*fs + 1 # total number of points (must be odd for symmetry)
        linspTorch1 = torch.linspace(-M, M, N, requires_grad=False, device=self.device)
        linspTorch = (linspTorch1 + tau * M)
        sinc1 = torch.sin(2 * np.pi * fc * linspTorch) / (np.pi * linspTorch)
        sinc1[torch.isnan(sinc1)] = 2*fc
        sinc = sinc1 / fs
        return sinc

    def sincForward(self, inputs):
        outputs = []
        for i in range(0, self.featSize):
            weight = self.MakeSincKernel(self.fc, self.fs, self.M, self.taus[i])
            theInput = inputs[:, :, :, i].view(inputs.size()[0], inputs.size()[1], inputs.size()[2], 1)
            output = torch.nn.functional.conv2d(theInput, weight.view(1, 1, self.kernel_size, 1))
            # from matplotlib import pyplot as plt
            # size = output.size()[2]
            # X = list(range(size))
            # plt.plot(X[:-self.shift],theInput.view(theInput.size()[2])[self.kernel_size-1:-self.shift].cpu().data.numpy(), color='red')
            # plt.plot(X[:-self.shift],output.view(output.size()[2])[self.shift:].cpu().data.numpy(), color='blue')
            # plt.legend(['ins', 'outs'], loc=6)
            # plt.show()
            outputs.append(output)#output[:, :, self.shift:, :]
        return torch.cat(outputs, dim=3)

    def forward(self, x):
        batch_size, timesteps, sq_len = x.size()
        x = x.view(batch_size, 1, timesteps, sq_len) # if sq_len more than one, it'll be used as input_channel
        pad = torch.nn.ConstantPad2d((0, 0, self.shift, self.shift), 0) # add padding to equalize size
        x = pad(x)
        output = self.sincForward(x)
        return output


class CCC(nn.Module):
    def __init__(self):
        super(CCC, self).__init__()

    def forward(self, prediction, ground_truth):
        # ground_truth = (ground_truth == torch.arange(self.num_classes).cuda().reshape(1, self.num_classes)).float()
        # ground_truth = ground_truth.squeeze(0)
        # prediction = prediction.view(prediction.size()[0]*prediction.size()[1])#.squeeze(1)
        # print("")
        # print("ground_truth", ground_truth.shape)
        # print("prediction", prediction.shape)
        mean_gt = torch.mean (ground_truth, 0)
        mean_pred = torch.mean (prediction, 0)
        var_gt = torch.var (ground_truth, 0)
        var_pred = torch.var (prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
        cov = torch.mean(v_pred * v_gt)
        numerator=2*cov
        ccc = numerator/denominator
        # print("ccc", ccc, mean_gt, mean_pred,var_gt,var_pred)
        return 1-ccc
        
class CrossCCC(nn.Module):
    def __init__(self, stride=1, maxStride=25*10):
        super(CrossCCC, self).__init__()
        self.stride = stride
        self.maxStride = maxStride

    def forward(self, prediction, ground_truth):
        # ground_truth = (ground_truth == torch.arange(self.num_classes).cuda().reshape(1, self.num_classes)).float()
        # ground_truth = ground_truth.squeeze(0)
        # prediction = prediction.view(prediction.size()[0]*prediction.size()[1])#.squeeze(1)
        # print("")
        # print("ground_truth", ground_truth.shape)
        # print("prediction", prediction.shape)
        CrossCCC = 0
        rng = range(0, self.maxStride, self.stride)
        CrossCCCs = []
        for n in rng:  
            newSize = prediction.size()[0]-n 
            pred = prediction[:-n] if n>0 else prediction
            pred = pred.view(1, 1, newSize, 1)
            pad = torch.nn.ConstantPad2d((0, 0, n, 0), 0)
            pred = pad(pred).view(prediction.size()[0])
            grd = ground_truth
            mean_gt = torch.mean (grd, 0)
            mean_pred = torch.mean(pred, 0)
            var_gt = torch.var (grd, 0)
            var_pred = torch.var(pred, 0)
            v_pred = pred - mean_pred
            v_gt = grd - mean_gt
            denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
            cov = torch.mean(v_pred * v_gt)
            numerator=2*cov
            ccc = numerator/denominator
            CrossCCC += ccc
            CrossCCCs.append(ccc)
        # print("C Min argMax Max Var", torch.min(torch.stack(CrossCCCs)), torch.argmax(torch.stack(CrossCCCs)), torch.max(torch.stack(CrossCCCs)), torch.var(torch.stack(CrossCCCs)))
        # print(CrossCCCs)
        # crosses = torch.stack(CrossCCCs)
        # from matplotlib import pyplot as plt
        # size = crosses.size()[0]
        # X = np.array(list(range(size)))/25#25hz sampling rate, turn into seconds
        # plt.plot(X,crosses.cpu().data.numpy(), color='red')
        # plt.legend(['crosses'], loc=1)
        # plt.show()

            # if n%10==0:
            #     from matplotlib import pyplot as plt
            #     size = pred.size()[0]
            #     X = list(range(size))
            #     plt.plot(X,pred.cpu().data.numpy(), color='red')
            #     plt.plot(X,grd.cpu().data.numpy(), color='blue')
            #     plt.legend(['tars', 'otars'], loc=6)
            #     plt.show()

            # allcs.append(ccc.item())
        # print(allcs)
        CrossCCC /= len(list(rng))
        # print("ccc", ccc, mean_gt, mean_pred,var_gt,var_pred)
        return 1-CrossCCC



class WindowCCC(nn.Module):
    def __init__(self, stride=1, maxStride=25*5, window_size=25*20):
        super(WindowCCC, self).__init__()
        self.stride = stride
        self.maxStride = maxStride
        self.window_size = int(window_size)

    def forward(self, prediction, ground_truth):
        # ground_truth = (ground_truth == torch.arange(self.num_classes).cuda().reshape(1, self.num_classes)).float()
        # ground_truth = ground_truth.squeeze(0)
        # prediction = prediction.view(prediction.size()[0]*prediction.size()[1])#.squeeze(1)
        # print("")
        # print("ground_truth", ground_truth.shape)
        # print("prediction", prediction.shape)
        rng = range(0, self.maxStride, self.stride)
        lngth = ground_truth.size()[0]
        num_wins = int(lngth/self.window_size)
        grd_wined = ground_truth[:num_wins*self.window_size].view(num_wins ,self.window_size)
        pred_wined = prediction[:num_wins*self.window_size].view(num_wins ,self.window_size)
        # allcs = []
        Cmins = 0
        for w in range(num_wins):
            pred = pred_wined[w]
            grd = grd_wined[w]
            CrossCCCs = []
            for n in rng:  
                newSize = pred.size()[0]-n 
                pred = pred[n:].view(1, 1, newSize, 1)
                pad = torch.nn.ConstantPad2d((0, 0, n, 0), 0)
                pred = pad(pred).view(self.window_size)
                mean_gt = torch.mean (grd, 0)
                mean_pred = torch.mean(pred, 0)
                var_gt = torch.var (grd, 0)
                var_pred = torch.var(pred, 0)
                v_pred = pred - mean_pred
                v_gt = grd - mean_gt
                denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
                cov = torch.mean(v_pred * v_gt)
                numerator=2*cov
                ccc = numerator/denominator
                CrossCCCs.append(ccc)
            Cmin = torch.min(torch.stack(CrossCCCs))
            Cmins += Cmin
        Cmins /= num_wins
            # allcs.append(ccc.item())
        # print(allcs)
        # CrossCCC /= len(list(rng))
        # print("ccc", ccc, mean_gt, mean_pred,var_gt,var_pred)
        return 1-Cmins