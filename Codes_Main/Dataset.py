import pandas as pd
import glob, os

class Dataset():
    """
    A class for loading data for machine learning purposes.
    This class only prepares data, thus how to read each file has to be provided via a function since it depends on the data format and has infinite possibilities.
    It is built upon the famous pandas package.

    ...

    Attributes
    ----------
    inputFilePaths : [string]
        The list containing the paths to the files containing the inputs.
    targetFilePaths : [string] (default = [""])
        The list containing the paths to the files containing the targets. [""] means there is no label (usually intended for testing).
    sampleNames : [string] (default = [""])
        The list containing the sample names for inputs and targets. [""] means the sample names are the same as the input file names without the extensions (e.g. "train_1.csv" -> "train_1").
    inputReader : function (default = lambda x: x)
        The function that reads the input files into the format ready for use.
    targetReader : function (default = lambda x: x)
        The function that reads the target files into the format ready for use.

    Methods
    -------
    initializeData()
        Initializes the paths data into a pandas dataframe.
    shuffle()
        shuffles the order of data of the pandas dataframe.
    __len__()
        returns the number of data.
    __getitem__(idx)
        returns (input, output) at the idx row.
    
    """
    def __init__(self, inputFilePaths:[str], targetFilePaths:[str]=[""], sampleNames:[str]=[""], 
                 inputReader=lambda fileName,sampleName="": sampleName, targetReader=lambda fileName,sampleName="": sampleName):
        self.inputFilePaths = inputFilePaths
        self.targetFilePaths = targetFilePaths
        self.sampleNames = sampleNames
        self.inputReader = inputReader
        self.targetReader = targetReader
        self.initializeData()

    def initializeData(self):
        if self.targetFilePaths == [""]: self.targetFilePaths = ["" for _ in self.inputFilePaths]
        if self.sampleNames == [""]: self.sampleNames = [filePath[filePath.rfind("/")+1:filePath.rfind(".")] for filePath in self.inputFilePaths]
        d = {'names':self.sampleNames, 'inputs':self.inputFilePaths, 'targets':self.targetFilePaths}
        self.dataFrame = pd.DataFrame(data=d)
        # print(self.dataFrame)

    def shuffle(self):
        self.dataFrame = self.dataFrame.sample(frac=1)

    def __len__(self):
        return self.dataFrame.shape[0]
    
    def __getitem__(self, idx):
        sampleName = self.dataFrame.iloc[idx].names
        inputFileName = self.dataFrame.iloc[idx].inputs
        targetFileName = self.dataFrame.iloc[idx].targets
        theInput = self.inputReader(inputFileName, sampleName=sampleName)
        theOutput = self.targetReader(targetFileName, sampleName=sampleName)
        return theInput, theOutput


# import sys; sys.path.append('.')
# inputFilePaths = glob.glob(os.path.join("./cases/RECOLA/inputs", "train_*.csv"), recursive=True)
# targetFilePaths = glob.glob(os.path.join("./cases/RECOLA/targets/valence", "train_*.csv"), recursive=True)
# inputFilePaths.sort()
# targetFilePaths.sort()
# Dataset = Dataset(inputFilePaths, targetFilePaths=targetFilePaths)
# print(len(Dataset))
# Dataset.shuffle()
# print(Dataset.dataFrame)
# print(Dataset[2])