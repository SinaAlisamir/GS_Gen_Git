import torch
import torch.nn as nn
from torch.autograd import Variable
from Models import CCC, SincModel
from Dataset import Dataset
from torch.utils.data import DataLoader
from funcs import printProgressBar, writeLineToCSV
import os, glob
import pandas as pd
import numpy as np

class ModelTrainer():
    def __init__(self, featureModel, labelModel, initStuff=True, run_name="run_2", device="cuda", inputFolder="", targetFolder="", listAnnots=[]):
        super(ModelTrainer, self).__init__()
        self.featureModel = featureModel
        self.labelModel = labelModel
        self.run_name = run_name
        self.device = device
        self.savePath = os.path.join(".", "runs", self.run_name)
        self.saveFilePath = os.path.join(self.savePath, "model.pth")
        self.saveFilePathOut = os.path.join(self.savePath, "modelOut.pth")
        self.trainOnAll = False
        self.inputFolder = inputFolder
        self.targetFolder = targetFolder
        self.listAnnots = listAnnots
        if initStuff: self.initLoadingScenario()

    def initLoadingScenario(self):
        self.batch_size = 1
        self.batch_size_eval = 1
        self.featSize = self.getDataset()[0][0].shape[1]
        # tausLabels = torch.zeros(1, device=self.device, requires_grad=False).float()
        self.model = self.featureModel
        self.model.to(device=self.device)
        self.modelOut = self.labelModel#LabelModel(6, fc=0.25, fs=25, M=10, pretaus=None)
        self.modelOut.to(device=self.device)
        self.criterion = CCC()
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.modelOut.parameters()), lr=0.001)

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
                outputs = self.model(inputs)
                # print(inputs.size(), targets.size())
                # origtars = targets[0,:,0]
                targets = self.modelOut(targets)
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

                eachLoss = 0
                tars = targets[:,:,:,0].view(targets.size()[0]*targets.size()[1]*targets.size()[2])
                outs = outputs[:,:,:,0].view(outputs.size()[0]*outputs.size()[1]*outputs.size()[2])
                loss = self.criterion(outs, tars) 
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

    def testModel(self, dataloader):
        self.model.eval()
        losses = np.array([])
        # print(self.model.taus*self.model.M)
        for (inputs, targets) in dataloader:
            inputs = inputs.to(device=self.device)
            targets = targets.to(device=self.device)
            # inputs, targets = targets, inputs
            # self.optimizer.zero_grad()
            # if len(inputs.size()) != 3: inputs = inputs.view(inputs.size()[0], inputs.size()[1], 1)
            outputs = self.model(inputs)
            targets = self.modelOut(targets)
            # tars = targets.view(targets.size()[0]*targets.size()[1]*targets.size()[2])
            # outs = outputs.view(outputs.size()[0]*outputs.size()[1]*outputs.size()[2])
            eachLoss = 0
            tars = targets[:,:,:,0].view(targets.size()[0]*targets.size()[1]*targets.size()[2])
            outs = outputs[:,:,:,0].view(outputs.size()[0]*outputs.size()[1]*outputs.size()[2])
            loss = self.criterion(outs, tars) 
            # loss.backward(retain_graph=True)
            # self.optimizer.step()
            losses = np.append(losses, loss.detach().cpu().numpy())
        return losses

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
        return dataset

    def getTrainDataLoader(self):
        dataloader = DataLoader(dataset=self.getDataset(prefix="train_"), batch_size=self.batch_size, shuffle=True)
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
        listAnnots = self.listAnnots
        # listAnnots = ['FM1 ', 'FM2 ', 'FM3 ', 'FF1 ', 'FF2 ', 'FF3']
        for annot in listAnnots:
            out = df[annot].to_numpy()
        # if standardize: out = (out - out.mean(axis=0)) / out.std(axis=0)
            out = np.expand_dims(out, axis=1)
            out = torch.FloatTensor(out)
            out = Variable(out)
            outs.append(out)
        outs = torch.cat(outs, 1)
        # print(outs.size())
        return outs
