'''

Emotion recognition based on generated annots

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
from Models import CCC, SincModel
from FeatureModels import *
from LabelModels import *
from ModelTrainer import ModelTrainer

def main():
    outputFolderArousal = "./Results/Emo_Pred-arousal"
    outputFolderValence = "./Results/Emo_Pred-valence"
    inputFolder = "../Data/mfb-mean" # audio_features_egemaps_0.04_mean / mfb-mean
    # targetFolderArousal = "./Results/27-10-arousal/GenAnnots"#"/media/sina/HD-Storage/Databases/RECOLA/all_ratings/arousal" # "/home/sina/Documents/Codes/DelayCompensation2/Results/13-0.7-0.3-arousal/GenAnnots"
    # targetFolderValence = "./Results/27-10-valence/GenAnnots"#"/media/sina/HD-Storage/Databases/RECOLA/all_ratings/valence" # "/home/sina/Documents/Codes/DelayCompensation2/Results/13-0.7-0.3-valence/GenAnnots"
    # targetFolderArousal = "/media/sina/HD-Storage/Databases/RECOLA/all_ratings/arousal" # "/home/sina/Documents/Codes/DelayCompensation2/Results/13-0.7-0.3-arousal/GenAnnots"
    # targetFolderValence = "/media/sina/HD-Storage/Databases/RECOLA/all_ratings/valence" # "/home/sina/Documents/Codes/DelayCompensation2/Results/13-0.7-0.3-valence/GenAnnots"

    if not os.path.exists(outputFolderArousal): os.makedirs(outputFolderArousal)
    if not os.path.exists(outputFolderValence): os.makedirs(outputFolderValence)

    featureSize = 40
    # FeatureModels = [FeatureModel0(featureSize), FeatureModel2(featureSize)]#, FeatureModel1(featureSize, kernel=(25,1)), FeatureModel2(featureSize)]
    FeatureModels = [FeatureModel2(featureSize)]
    LabelModels   = [LabelModel1(1)]

    # coefs = [[1.9, 0.1], [1.8, 0.2], [1.7, 0.3], [1.6, 0.4], [1.5, 0.5], [1.4, 0.6], [1.3, 0.7], [1.2, 0.8], [1.1, 0.9], [1.0, 1.0],
    #          [0.9, 1.1], [0.8, 1.2], [0.7, 1.3], [0.6, 1.4], [0.5, 1.5], [0.4, 1.6], [0.3, 1.7], [0.2, 1.8], [0.1, 1.9]]
    # coefs = [[0.9, 1.1], [0.8, 1.2], [0.7, 1.3], [0.6, 1.4], [0.5, 1.5], [0.4, 1.6], [0.3, 1.7], [0.2, 1.8], [0.1, 1.9]]
    coefs = [[1.8, 0.2]]
    for coef in coefs:
        print(coef)
        targetFolderArousal = "./Results/GS_Gen-"+str(coef[0])+"-arousal/GenAnnots"
        targetFolderValence = "./Results/GS_Gen-"+str(coef[0])+"-valence/GenAnnots"
        for i, targetFolder in enumerate([targetFolderArousal, targetFolderValence]):
            task = "Arousal" if i==0 else "Valence"
            # if i==0:continue
            for featureModel in FeatureModels:
                for labelModel in LabelModels:
                    run_specify = "_" + task + "_" + str(featureModel.__class__.__name__) + "_" + str(labelModel.__class__.__name__) + "_GS_Gen-"+str(coef)
                    outputFolder = outputFolderArousal if i==0 else outputFolderValence

                    headers = ["Annots"]
                    listAnnots = ['GoldStandard']
                    annot = "GoldStandard"
                    values = [annot]

                    headers += ["run_specify"]
                    values += [run_specify]

                    modelTrainer = ModelTrainer(featureModel, labelModel, run_name=os.path.join("run_Emo_Pred", run_specify), inputFolder=inputFolder, targetFolder=targetFolder, listAnnots=listAnnots)
                    evalLosses = modelTrainer.trainModel(maxEpoch = 150, tolerance = 10, printLevel=1, trainOnAll=False)
                    modelTrainer.loadModel()

                    modelTrainer.batch_size_eval = 1
                    losses = modelTrainer.testModel(modelTrainer.getTrainDataLoader())
                    CCCs = 1 - losses
                    CCCmean = np.mean(CCCs)
                    CCCvar = np.var(CCCs)
                    headers += ["CCC mean train"]
                    values += [str(np.round(CCCmean, 3))]
                    headers += ["CCC var train"]
                    values += [str(np.round(CCCvar, 3))]

                    modelTrainer.batch_size_eval = 9
                    lossesAll = modelTrainer.testModel(modelTrainer.getTrainDataLoader())
                    CCCsAll = 1 - lossesAll
                    CCCmeanAll = np.mean(CCCsAll)
                    CCCvarAll = np.var(CCCsAll)
                    headers += ["CCC mean train All"]
                    values += [str(np.round(CCCmeanAll, 3))]
                    headers += ["CCC var train All"]
                    values += [str(np.round(CCCvarAll, 3))]

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

                    # modelTrainer.batch_size_eval = 9
                    # lossesAll = modelTrainer.testModel(modelTrainer.getTestDataLoader())
                    # CCCsAll = 1 - lossesAll
                    # CCCmeanAll = np.mean(CCCsAll)
                    # CCCvarAll = np.var(CCCsAll)
                    # headers += ["CCC mean test All"]
                    # values += [str(np.round(CCCmeanAll, 3))]
                    # headers += ["CCC var test All"]
                    # values += [str(np.round(CCCvarAll, 3))]

                    try:
                        taus = modelTrainer.modelOut.sinc.taus.detach().cpu().numpy()
                        for t, tau in enumerate(taus):
                            tau = tau * modelTrainer.modelOut.sinc.M
                            headers += ["tau " + str(t)]
                            values += [tau]
                    except:
                        pass   

                    try:
                        alltaus = modelTrainer.modelOut.allSinc.taus.detach().cpu().numpy()
                        for t, tau in enumerate(alltaus):
                            tau = tau * modelTrainer.modelOut.sinc.M
                            headers += ["alltau " + str(t)]
                            values += [tau]
                    except:
                        pass
                    
                    try:
                        weights = modelTrainer.modelOut.softmax(modelTrainer.modelOut.weights).detach().cpu().numpy()
                        for t, weight in enumerate(weights):
                            headers += ["weight " + str(t)]
                            values += [weight]
                    except:
                        pass

                    print(CCCmean, CCCvar)
                    
                    writeLineToCSV(os.path.join(outputFolder, "results.csv"), headers, values)

if __name__ == "__main__":
    main()
