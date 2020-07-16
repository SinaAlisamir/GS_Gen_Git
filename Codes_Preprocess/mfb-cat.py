import os
from funcs import get_files_in_path
from funcs import printProgressBar
from python_speech_features import logfbank
import librosa
import numpy as np
import pandas as pd

class mfb_from_wav():
    """
    Calculate mel-frequency filterbank features from wav files.

    ...

    Attributes
    ----------
    WavsFolder : string
        The path to the folder containing wav files.
    csvOutFolder : string
        The path to the folder containing csv files for the filterbank features. If doesn't exist, it will get created.
    sr : integer (default = 16000)
        The sampling rate (Hz) of the wav files.
        

    Methods
    -------
    calc_mfb(concat=0)
        Calculate the mfb feats for WavsFolder and save them under csvOutFolder.
        concat will concatenate inputs together. If 2 for example, each 2 inputs near each other will get concatenated to form a new vector.
        by default = 0, it doesn't get activated!
    
    """
    def __init__(self, wavsFolder:str, csvOutFolder:str, sr=16000, normalized=True):
        self.wavsFolder = get_files_in_path(wavsFolder)
        self.csvOutFolder = csvOutFolder
        self.normalized = normalized
        self.sr = sr
        
    def calc_mfbs(self, concat=0):
        self.makePath(self.csvOutFolder)
        for p, path in enumerate(self.wavsFolder):
            sig, rate = librosa.load(path, sr=self.sr)
            # print(sig.shape, rate)
            fbank_feat = logfbank(sig, rate, winlen=0.025, winstep=0.01, nfilt=40, nfft=2028, lowfreq=0, highfreq=None, preemph=0.97) #, winfunc=np.hanning
            if self.normalized: fbank_feat = (fbank_feat - fbank_feat.mean(axis=0)) / fbank_feat.std(axis=0)
            if concat > 0: fbank_feat = self.concatVecs(fbank_feat, concat)
            # print(fbank_feat.shape)
            # np.savetxt(os.path.join(self.csvOutFolder, self.getFileNameAndChangeExt(path)), fbank_feat, delimiter=',')
            header = ['mfb '+str(i) for i in range(len(fbank_feat[0]))]
            df = pd.DataFrame(data=fbank_feat,columns=header)
            outPath = os.path.join(self.csvOutFolder, self.getFileNameAndChangeExt(path))
            with open(outPath, 'w', encoding='utf-8') as f:
                df.to_csv(f, index=False)
            printProgressBar(p + 1, len(self.wavsFolder), prefix = 'Calculating MFBs:', suffix = 'Complete', length = 100)
    
    @staticmethod
    def concatVecs(fbank_feat, concat):
        # print(fbank_feat.shape)
        feat_size = fbank_feat.shape[1]
        new_feat_size = fbank_feat.shape[1]*concat
        new_num_feats = fbank_feat.shape[0]//concat
        new_fbank_feat = fbank_feat[:new_num_feats*concat].reshape((new_num_feats, new_feat_size))
        return new_fbank_feat
                
    @staticmethod
    def getFileNameAndChangeExt(filePath, ext=".csv"):
        return filePath[filePath.rfind("/")+1:filePath.rfind(".")] + ext

    @staticmethod
    def makePath(csvPath):
        if not os.path.exists(csvPath): os.makedirs(csvPath)

if __name__== "__main__":
    import sys
    print (sys.argv)
    
    # wavPath = sys.argv[1]#"./tests/fixtures/audios"
    # csvPath = sys.argv[2]#"./tests/fixtures/csvs"
    wavPath = "../Data/recordings_audio"
    csvPath = "../Data/mfb-cat"
    mfbExtractor = mfb_from_wav(wavPath, csvPath, 44100)
    mfbExtractor.calc_mfbs(concat=4)
