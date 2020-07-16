'''

Modeling the annots with RNNs! generating new gold-standard from annotations
one BGRU for all annots then average weighting
Lin on feats

'''

from Dataset import Dataset
from torch.utils.data import DataLoader
from funcs import printProgressBar, writeLineToCSV
import os, glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Models import CCC, SincModel, CrossCCC
# from scipy.signal import savgol_filter

def main():
    inputFolder = "../Data/mfb-mean" # audio_features_egemaps_0.04_mean / mfb-mean
    targetFolderArousal = "../Data/all_ratings/arousal"
    targetFolderValence = "../Data/all_ratings/valence" # /media/sina/HD-Storage/Databases/RECOLA


    df = pd.read_csv(os.path.join(inputFolder, "train_1.csv"), delimiter=",")
    feats = list(df.columns)
    # feats.remove("name"); feats.remove("frameTime")

    # coefs = [[1.9, 0.1], [1.85, 0.15], [1.825, 0.175], [1.8, 0.2], [1.7, 0.3], [1.6, 0.4], [1.5, 0.5], [1.4, 0.6], [1.3, 0.7], [1.2, 0.8], [1.1, 0.9], [1.0, 1.0]]
    coefs = [[1.8, 0.2]]
    # coefs = [[0.9, 1.1], [0.8, 1.2], [0.7, 1.3], [0.6, 1.4], [0.5, 1.5], [0.4, 1.6], [0.3, 1.7], [0.2, 1.8], [0.1, 1.9]]
    for coef in coefs:
        print(coef)
        outputFolderArousal = "./Results/GS_Gen-"+str(coef[0])+"-arousal"
        outputFolderValence = "./Results/GS_Gen-"+str(coef[0])+"-valence"
        if not os.path.exists(outputFolderArousal): os.makedirs(outputFolderArousal)
        if not os.path.exists(outputFolderValence): os.makedirs(outputFolderValence)
        for i, targetFolder in enumerate([targetFolderArousal, targetFolderValence]):
            outputFolder = outputFolderArousal if i==0 else outputFolderValence
            print("Arousal") if i==0 else print("Valence")
            # if i==0: continue
            headers = ["Annots"]
            annot = "GoldStandard"
            values = [annot]
            run_specify = annot + "_" + str(i) + "_" + str(coef[0])

            modelTrainer = ModelTrainer(run_name=os.path.join("run_GS_Gen", run_specify), l_rate=0.01, inputFolder=inputFolder, targetFolder=targetFolder, coef=coef)
            evalLosses = modelTrainer.trainModel(maxEpoch = 150, tolerance = 10, printLevel=1, trainOnAll=False)
            modelTrainer.loadModel()

            modelTrainer.batch_size_eval = 1
            losses = modelTrainer.testModel(modelTrainer.getDevDataLoader())
            CCCs = 1 - losses
            CCCmean = np.mean(CCCs)
            CCCvar = np.var(CCCs)
            headers += ["CCC mean dev"]
            values += [str(np.round(CCCmean, 3))]
            headers += ["CCC var dev"]
            values += [str(np.round(CCCvar, 3))]

            modelTrainer.batch_size_eval = 9
            lossesAll = modelTrainer.testModel(modelTrainer.getDevDataLoader())
            CCCsAll = 1 - lossesAll
            CCCmeanAll = np.mean(CCCsAll)
            CCCvarAll = np.var(CCCsAll)
            headers += ["CCC mean dev All"]
            values += [str(np.round(CCCmeanAll, 3))]
            headers += ["CCC var dev All"]
            values += [str(np.round(CCCvarAll, 3))]

            # modelTrainer.batch_size_eval = 1
            # losses = modelTrainer.testModel(modelTrainer.getTestDataLoader())
            # CCCs = 1 - losses
            # CCCmean = np.mean(CCCs)
            # CCCvar = np.var(CCCs)
            # headers += ["CCC mean test"]
            # values += [str(np.round(CCCmean, 3))]
            # headers += ["CCC var test"]
            # values += [str(np.round(CCCvar, 3))]

            modelTrainer.batch_size_eval = 9
            lossesAll, lossesAnnotAll, lossesLLDAll = modelTrainer.testModel(modelTrainer.getDevDataLoader(), otherLosses=True)
            CCCsAll = 1 - lossesAll
            CCCmeanAll = np.mean(CCCsAll)
            CCCvarAll = np.var(CCCsAll)
            headers += ["CCC mean dev All"]
            values += [str(np.round(CCCmeanAll, 3))]
            headers += ["CCC var dev All"]
            values += [str(np.round(CCCvarAll, 3))]
            CCCsAll = 1 - lossesAnnotAll
            CCCmeanAll = np.mean(CCCsAll)
            CCCvarAll = np.var(CCCsAll)
            headers += ["CCC Annot mean dev All"]
            values += [str(np.round(CCCmeanAll, 3))]
            headers += ["CCC Annot var dev All"]
            values += [str(np.round(CCCvarAll, 3))]
            CCCsAll = 1 - lossesLLDAll
            CCCmeanAll = np.mean(CCCsAll)
            CCCvarAll = np.var(CCCsAll)
            headers += ["CCC feat mean dev All"]
            values += [str(np.round(CCCmeanAll, 3))]
            headers += ["CCC feat var dev All"]
            values += [str(np.round(CCCvarAll, 3))]

            print(CCCmean, CCCvar)
            writeLineToCSV(os.path.join(outputFolder, "results.csv"), headers, values)

            # from matplotlib import pyplot as plt
            # modelTrainer.batch_size_eval = 1
            # dataloader = modelTrainer.getTestDataLoader()
            # for (inputs, targets) in dataloader:
            #     inputs = inputs.to(device=modelTrainer.device)
            #     targets = targets.to(device=modelTrainer.device)
            #     origtars = targets
            #     outputs = modelTrainer.model(inputs)
            #     outs = outputs[:,:,:].view(outputs.size()[0]*outputs.size()[1]*outputs.size()[2])
            #     targets, outputsRNN1, outputsRNN2 = modelTrainer.modelOut(torch.cat((targets, inputs), dim=2))
            #     ins = inputs[:,:,0].view(inputs.size()[0]*inputs.size()[1])
            #     ins /= torch.max(ins)
            #     targs = targets.squeeze().squeeze().squeeze()
            #     tars = outputsRNN1[:,:,0].view(outputsRNN1.size()[0]*outputsRNN1.size()[1])
            #     tars1 = outputsRNN1[:,:,4].view(outputsRNN1.size()[0]*outputsRNN1.size()[1])
            #     otars = origtars[:,:,0].view(origtars.size()[0]*origtars.size()[1])
            #     size = tars.size()[0]
            #     X = list(range(size))
            #     # print('Weights', modelTrainer.modelOut.softmax(modelTrainer.modelOut.weights))
            #     targsFiltered = targs.cpu().data.numpy()#getSmooth(targs.cpu().data.numpy(), win=smooth)
            #     print(1-modelTrainer.criterionC(tars, otars), 1-modelTrainer.criterion(outs, targs))
            #     # print("2", 1-modelTrainer.criterionC(torch.tensor(getSmooth(tars.cpu().data.numpy(), win=smooth)).cuda(), otars))
            #     print("3", 1-modelTrainer.criterion(outs, torch.tensor(targsFiltered).cuda()))
            #     # print("4", 1-modelTrainer.criterion(otars, ins), 1-modelTrainer.criterionC(otars, ins))
            #     # plt.plot(X,tars.cpu().data.numpy(), color='red')
            #     # plt.plot(X,tars1.cpu().data.numpy(), color='black')
            #     plt.plot(X,otars.cpu().data.numpy(), color='blue')
            #     plt.plot(X,ins.cpu().data.numpy(), color='green')
            #     plt.plot(X,targsFiltered, color='purple')
            #     plt.legend(['original GS', 'feature', 'targs'], loc=1)
            #     plt.show()

            modelTrainer.batch_size_eval = 1
            dataloader = modelTrainer.getTrainDataLoader(shuffle=False)
            modelTrainer.printOutCSV(dataloader, os.path.join(outputFolder, "GenAnnots"), prefix='train_')
            dataloader = modelTrainer.getDevDataLoader()
            modelTrainer.printOutCSV(dataloader, os.path.join(outputFolder, "GenAnnots"), prefix='dev_')
            # dataloader = modelTrainer.getTestDataLoader()
            # modelTrainer.printOutCSV(dataloader, os.path.join(outputFolder, "GenAnnots"), prefix='test_')


class ModelTrainer():
    def __init__(self, initStuff=True, run_name="run_2", l_rate=0.01, device="cuda", inputFolder="", targetFolder="", coef=[1,1]):
        super(ModelTrainer, self).__init__()
        self.run_name = run_name
        self.device = device
        self.l_rate = l_rate
        self.coef = coef
        self.listAnnots = ['FM1 ', 'FM2 ', 'FM3 ', 'FF1 ', 'FF2 ','FF3']
        self.savePath = os.path.join(".", "runs", self.run_name)
        self.saveFilePath = os.path.join(self.savePath, "model.pth")
        self.saveFilePathOut = os.path.join(self.savePath, "modelOut.pth")
        self.trainOnAll = False
        self.inputFolder = inputFolder
        self.targetFolder = targetFolder
        if initStuff: self.initLoadingScenario()

    def initLoadingScenario(self):
        self.batch_size = 1
        self.batch_size_eval = 1
        self.featSize = self.getDataset()[0][0].shape[1]
        # tausLabels = torch.zeros(1, device=self.device, requires_grad=False).float()
        self.model = FeatureModel(self.featSize)
        self.model.to(device=self.device)
        self.modelOut = LabelModel(6+self.featSize, fc=0.25, fs=25, M=10, pretaus=None)
        self.modelOut.to(device=self.device)
        self.criterion = CCC()
        self.criterionC = CrossCCC(stride=1, maxStride=25*5)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.modelOut.parameters()), lr=self.l_rate)

    def trainModel(self, maxEpoch = 100, tolerance = 15, printLevel=1, trainOnAll=False):
        evalslossMeans = []
        self.trainOnAll = trainOnAll
        dataloader = self.getTrainDataLoader()
        dataloaderDev = self.getDevDataLoader()
        for epoch in range(1, maxEpoch+1):
            self.model.train()
            losses = np.array([])
            for (inputs, targets) in dataloader:
                inputs = inputs.to(device=self.device)
                targets = targets.to(device=self.device)
                # inputs, targets = targets, inputs
                self.optimizer.zero_grad()
                # print("inputs.size()", inputs.size())
                # if len(inputs.size()) != 3: inputs = inputs.view(inputs.size()[0], inputs.size()[1], 1)
                # print(inputs.size(), targets.size(), torch.cat((targets, inputs), dim=2).size())
                outputs = self.model(inputs)
                # print(inputs.size(), targets.size())
                # origtars = targets[0,:,0]
                origtars = targets
                targets, outputsRNN1, outputsRNN2 = self.modelOut(torch.cat((targets, inputs), dim=2))
                # print(outputs.size(), targets.size())
                # origtars = targets.view(targets.size()[0]*targets.size()[1]*targets.size()[2])[:-self.modelOut.shift]
                # tars = targets.view(targets.size()[0]*targets.size()[1]*targets.size()[2])
                # outs = outputs.view(outputs.size()[0]*outputs.size()[1]*outputs.size()[2])

                # from matplotlib import pyplot as plt
                # tars = targets[:,:,:,0].view(targets.size()[0]*targets.size()[1]*targets.size()[2])
                # size = origtars.size()[0]
                # X = list(range(size))
                # plt.plot(X,tars.cpu().data.numpy(), color='red')
                # plt.plot(X,origtars.cpu().data.numpy(), color='blue')
                # plt.legend(['tars', 'origtars'], loc=6)
                # plt.show()

                tars = targets[:,:,:,0].view(targets.size()[0]*targets.size()[1]*targets.size()[2])
                outs = outputs[:,:,:].view(outputs.size()[0]*outputs.size()[1]*outputs.size()[2])
                # ins = inputs[:,:,:].view(inputs.size()[0]*inputs.size()[1]*inputs.size()[2])
                # tars2 = outputsRNN2[:,:,:].contiguous().view(outputsRNN2.size()[0]*outputsRNN2.size()[1]*outputsRNN2.size()[2])
                lossLLD = self.criterion(outs, tars)
                # lossFeats = self.criterion(ins, tars2)
                eachLoss = 0
                for o in range(outputsRNN1.size()[2]):
                    eachLoss += self.criterionC(outputsRNN1[:,:,o].view(outputsRNN1.size()[0]*outputsRNN1.size()[1]), origtars[:,:,o].view(origtars.size()[0]*origtars.size()[1]))
                lossAnnot = eachLoss / outputsRNN1.size()[2]

                loss = self.coef[0]*lossAnnot + self.coef[1]*lossLLD# + 0.2*lossFeats

                loss.backward(retain_graph=True)
                self.optimizer.step()
                losses = np.append(losses, loss.detach().cpu().numpy())

            evalLosses = self.testModel(dataloaderDev)
            evalLoss = np.mean(evalLosses)
            if printLevel == 1:
                printProgressBar(epoch + 1, maxEpoch + 1, prefix = 'Training Model:', suffix = 'Complete, CCC for dev: ' + str(round(1-evalLoss,3)) , length = 100)

            evalslossMeans.append(np.mean(evalLosses))
            writeLineToCSV(os.path.join(self.savePath, "trainLog.csv"), ["epoch", "evalLossMean"], [epoch, np.mean(evalLosses)])
            if min(evalslossMeans) == evalLoss: self.saveModel()
            bestEpoch = evalslossMeans.index(min(evalslossMeans))+1
            if (len(evalslossMeans) - bestEpoch >= tolerance) and epoch > 20:
                print("\nStopped early. Best loss was at epoch", bestEpoch, min(evalslossMeans))
                break

        # print("losses", losses, len(losses))
        # tau = self.model.sincConv.M * self.model.sincConv.taus[0]
        # return np.array(losses), tau.item()

    def testModel(self, dataloader, otherLosses=False, smooth=0):
        self.model.eval()
        losses = np.array([])
        lossesAnnot = np.array([])
        lossesLLD = np.array([])
        # print(self.model.taus*self.model.M)
        for (inputs, targets) in dataloader:
            inputs = inputs.to(device=self.device)
            targets = targets.to(device=self.device)
            # inputs, targets = targets, inputs
            # self.optimizer.zero_grad()
            # if len(inputs.size()) != 3: inputs = inputs.view(inputs.size()[0], inputs.size()[1], 1)
            outputs = self.model(inputs)
            origtars = targets
            targets, outputsRNN1, outputsRNN2 = self.modelOut(torch.cat((targets, inputs), dim=2))
            targets, outputsRNN1, outputsRNN2 = targets.detach(), outputsRNN1.detach(), outputsRNN2.detach()
            # tars = targets.view(targets.size()[0]*targets.size()[1]*targets.size()[2])
            # outs = outputs.view(outputs.size()[0]*outputs.size()[1]*outputs.size()[2])
            tars = targets[:,:,:,0].view(targets.size()[0]*targets.size()[1]*targets.size()[2])
            outs = outputs[:,:,:].view(outputs.size()[0]*outputs.size()[1]*outputs.size()[2])
            # ins = inputs[:,:,:].view(inputs.size()[0]*inputs.size()[1]*inputs.size()[2])
            # tars2 = outputsRNN2[:,:,:].contiguous().view(outputsRNN2.size()[0]*outputsRNN2.size()[1]*outputsRNN2.size()[2])

            if smooth > 0: tars = torch.Tensor(getSmooth(tars.cpu().data.numpy(), win=smooth)).cuda()
            lossLLD = self.criterion(outs, tars)
            # lossFeats = self.criterion(ins, tars2)
            eachLoss = 0
            for o in range(outputsRNN1.size()[2]):
                eachLoss += self.criterionC(outputsRNN1[:,:,o].view(outputsRNN1.size()[0]*outputsRNN1.size()[1]), origtars[:,:,o].view(origtars.size()[0]*origtars.size()[1]))
            lossAnnot = eachLoss / outputsRNN1.size()[2]

            loss = self.coef[0]*lossAnnot + self.coef[1]*lossLLD# + 0.2*lossFeats
            # loss.backward(retain_graph=True)
            # self.optimizer.step()
            losses = np.append(losses, loss.detach().cpu().numpy())
            lossesAnnot = np.append(lossesAnnot, lossAnnot.detach().cpu().numpy())
            lossesLLD = np.append(lossesLLD, lossLLD.detach().cpu().numpy())
        # print(np.mean(lossesAnnot), np.mean(lossesLLD))
        if otherLosses: losses = (losses, lossesAnnot, lossesLLD)
        return losses

    def printOutCSV(self, dataloader, savePath, prefix='train_', smooth=0):
        if prefix=="train_": tarPaths = self.targetFilePathsTrain
        if prefix=="dev_": tarPaths = self.targetFilePathsDev
        if prefix=="test_": tarPaths = self.targetFilePathsTest
        self.model.eval()
        headers = ['GoldStandard']
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device=self.device)
            targets = targets.to(device=self.device)
            targets, outputsRNN1, outputsRNN2 = self.modelOut(torch.cat((targets, inputs), dim=2))
            targets, outputsRNN1, outputsRNN2 = targets.detach(), outputsRNN1.detach(), outputsRNN2.detach()
            featSize = targets.size()[2]
            tars = targets.view(targets.size()[0]*targets.size()[1]*targets.size()[2]*targets.size()[3]).cpu().data.numpy()
            if smooth > 0: tars = getSmooth(tars, win=smooth)
            values = tars.reshape(tars.shape[0], 1)
            # print(targets.size(), values.shape)
            # for o in range(featSize):
            #     print(targets.size())
            #     tars = targets[:,:,o].view(targets.size()[0]*targets.size()[1])
            #     value = tars.cpu().data.numpy()
            #     value = value.reshape(value.shape[0], 1)
            #     values = np.concatenate((values, value), 1)
            df = pd.DataFrame(data=values,columns=headers)
            filePath = tarPaths[i]
            LastSlashPos = filePath.rfind(os.path.split(filePath)[-1])
            fileName = filePath[LastSlashPos:]
            if not os.path.exists(savePath): os.makedirs(savePath)
            with open(os.path.join(savePath, fileName), 'w', encoding='utf-8') as f:
                df.to_csv(f, index=False)

    def loadModel(self):
        self.model = torch.load(self.saveFilePath, map_location=self.device)
        self.modelOut = torch.load(self.saveFilePathOut, map_location=self.device)

    def saveModel(self):
        LastSlashPos = self.saveFilePath.rfind(os.path.split(self.saveFilePath)[-1]) - 1
        if not os.path.exists(self.saveFilePath[:LastSlashPos]): os.makedirs(self.saveFilePath[:LastSlashPos])
        torch.save(self.model, self.saveFilePath)
        torch.save(self.modelOut, self.saveFilePathOut)

    def getDataset(self, prefix="train_"):
        if self.trainOnAll == True: prefix=""
        inputFilePaths = glob.glob(os.path.join(self.inputFolder, prefix+"*.csv"), recursive=True)
        targetFilePaths = glob.glob(os.path.join(self.targetFolder, prefix+"*.csv"), recursive=True)
        inputFilePaths.sort()
        targetFilePaths.sort()
        dataset = Dataset(inputFilePaths, targetFilePaths=targetFilePaths, inputReader=self.inputReader, targetReader=self.targetReader)
        if prefix=="train_": self.inputFilePathsTrain = inputFilePaths; self.targetFilePathsTrain = targetFilePaths
        if prefix=="dev_": self.inputFilePathsDev = inputFilePaths; self.targetFilePathsDev = targetFilePaths
        if prefix=="test_": self.inputFilePathsTest = inputFilePaths; self.targetFilePathsTest = targetFilePaths
        return dataset

    def getTrainDataLoader(self, shuffle=True):
        dataloader = DataLoader(dataset=self.getDataset(prefix="train_"), batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def getDevDataLoader(self):
        dataloader = DataLoader(dataset=self.getDataset(prefix="dev_"), batch_size=self.batch_size_eval, shuffle=False)
        return dataloader

    def getTestDataLoader(self):
        dataloader = DataLoader(dataset=self.getDataset(prefix="test_"), batch_size=self.batch_size_eval, shuffle=False)
        return dataloader

    def inputReader(self, inputFileName, sampleName="", delimiter=",", standardize=True):
        """
        Reads recola input csv file for sample name.
        """
        df = pd.read_csv(inputFileName, delimiter=delimiter)
        # print("testé", df.to_numpy()[:,2:])
        # print(df.to_numpy()[:,2:].shape)
        # out = df["Loudness_sma3"].to_numpy()
        # if standardize: out = (out - out.mean(axis=0)) / out.std(axis=0)
        out = df.to_numpy().astype(float)#[:,2:]
        # print(out.mean(axis=0).shape)
        if standardize:
            for i in range(out.shape[1]): out[:,i] = (out[:,i] - out[:,i].mean(axis=0)) / out[:,i].std(axis=0)
        while out.shape[0] < 7501: out = np.append(out, np.array([out[-1]]), axis=0)
        # print(out)
        out = torch.FloatTensor(out)
        out = Variable(out)
        return out

    def targetReader(self, targetFileName, sampleName="", standardize=True):
        """
        Reads recola target csv file for sample name.
        """
        df = pd.read_csv(targetFileName)
        # out = df[self.columnNameTarget].to_numpy()
        outs = []
        for annot in self.listAnnots:
            out = df[annot].to_numpy()
        # if standardize: out = (out - out.mean(axis=0)) / out.std(axis=0)
            out = np.expand_dims(out, axis=1)
            out = torch.FloatTensor(out)
            out = Variable(out)
            outs.append(out)
        outs = torch.cat(outs, 1)
        # print(outs.size())
        return outs


class FeatureModel(nn.Module):
    def __init__(self, featSize, fc=12.5, fs=25, M=10, device="cuda"):
        super(FeatureModel, self).__init__()
        self.featSize = featSize
        self.device = device
        # self.sinc = SincModel(featSize, fc=0.25, fs=fs, M=M, device=device)
        # self.rnn = nn.GRU(
        #     input_size=featSize,
        #     hidden_size=128,
        #     # bidirectional=True,
        #     num_layers=2,
        #     batch_first=True)
        # self.out = nn.Linear(128, 1)
        self.lin = nn.Sequential(
            nn.Linear(featSize, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # batch_size, timesteps, sq_len = x.size()
        # x = self.sinc(x)
        # x = x.view(batch_size, timesteps, sq_len)
        # output, _ = self.rnn(x)
        # output = self.out(output)
        # output = output.view(batch_size, 1, timesteps, 1)

        return self.lin(x)

class LabelModel(nn.Module):
    def __init__(self, featSize, fc=5, fs=25, M=10, device="cuda", pretaus=None):
        super(LabelModel, self).__init__()
        self.featSize = featSize
        self.device = device
        # pretaus = torch.zeros(featSize, device=self.device, requires_grad=False).float()
        # self.sinc = SincModel(featSize, fc=fc, fs=fs, M=M, device=device, pretaus=None)
        # self.weights = torch.nn.Parameter(torch.rand(featSize, device=self.device, requires_grad=True).float())
        self.weights = torch.nn.Parameter(torch.ones(6, device=self.device, requires_grad=True).float())/6
        # print("self.weights", self.weights)
        self.softmax = nn.Softmax(dim=0)
        # self.allSinc = SincModel(1, fc=fc, fs=fs, M=M, device=device, pretaus=None)
        self.rnn = nn.GRU(input_size=featSize,hidden_size=16,num_layers=1,bidirectional=True,batch_first=True)
        self.out = nn.Linear(16*2, featSize)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print('x',x.size())
        batch_size, timesteps, sq_len = x.size()

        # weights = self.softmax(self.weights)
        # new_gsB = x * weights
        # print(new_gsB, new_gsB.size())
        # new_gs = torch.mean(new_gsB, 2).unsqueeze(2)

        output, _ = self.rnn(x)
        output = self.out(output)
        output = self.tanh(output)
        # print("output.size", output.size())
        outputRNN1, outputRNN2 = output[:,:,:6], output[:,:,6:]

        # output = self.sinc(outputRNN)
        # print(output.size())
        # output = torch.mean(output, 3).unsqueeze(3)
        
        weights = self.softmax(self.weights)
        output = outputRNN1 * weights
        output = torch.sum(output, 2).unsqueeze(2)

        # print(self.softmax(self.weights))
        # output = torch.sum(output, 2).unsqueeze(2)
        # print(output.size())
        # output = self.allSinc(output.view(batch_size, timesteps, 1))
        # batch_size, timesteps, sq_len = x.size()
        output = output.view(batch_size, 1, timesteps, 1)
        return output, outputRNN1, outputRNN2

def getSmooth(sig, win=25*1):
    mysig = sig.copy()
    aux = int(win/2)
    for i in range(aux, len(mysig)-aux):
        value = np.mean(sig[i-aux:i+aux])
        mysig[i] = value
    for i in range(1, aux):
        mysig[i] = np.mean(sig[:i])
    for i in range(len(mysig)-aux, len(mysig)):
        mysig[i] = np.mean(sig[i:])
    return mysig

if __name__ == "__main__":
    main()
