# Linearly-Constrained Generation of Dimensional Emotion Labels from Human Annotations with Recurrent Neural Networks

Paper explaining the concept:

(Submitted, waiting for acceptance ... or not!)

![NewGS](https://github.com/SinaAlisamir/GS_Gen_Git/blob/master/NewGS.png)

Dependencies:

```
python_speech_features 0.5 (only for preprocessing)
librosa 0.7.2 (only for preprocessing)
pandas 1.0.5
pytorch 1.4.0
```

Install dependencies from the `.yml` file:

```bash
conda env create -f env.yml
```



## Folder and files explanation

`Codes_Main`: A folder containing all the files relating to the main neural network based models used on the paper.

​	`Dataset.py`: This file is used for loading the data dynamically. We avoid loading all the data on memory, instead we load only the addresses referencing the files and the data would get loaded on memory if and when needed.

​	`FeatureModels.py`: This file consists of some `Pytorch` models used for emotion prediction.

​	`funcs.py`: This file consists of some generic functions used in different files.

​	`Models.py`: This file consists of some `Pytorch` models, specifically the ones related to `CCC` and `Cross-CCC`.

​	`ModelTrainer.py`: This file consists of a class used for training and testing `Pytorch` models.

​	`run_GS_Gen.py`: The main file for generating the new gold-standard. After execution the new gold-standard will be written under `Codes_Main/Results/GS_Gen-[alpha]-[arousal/valence]/GenAnnots`

​	`run_GS_GenS.py`: The main file for generating the smoothed new gold-standard. It uses the trained model from `run_GS_Gen.py` (this can be changed from the code). After execution the new gold-standard will be written under `Codes_Main/Results/GS_GenS-[alpha]-[arousal/valence]/GenAnnots`

​	`run_Emo_Pred.py`: The main file for the emotion prediction model from the new gold-standard.

​	`run_Emo_PredS.py`: The main file for the emotion prediction model from the new smoothed gold-standard.



`Codes_Preprocess`: A folder containing the files relating to the preprocessing and getting the features from audio. `recordings_audio` folder should exists under `Data` folder containing wav files from `RECOLA` dataset for feature extraction scripts.

​	`mfb-cat.py`: Log mel filter bank feature extraction (concatenates each 4 adjacent features).

​	`mfb-mean.py`: Log mel filter bank feature extraction (averages each 4 adjacent features).



`Data`: A folder containing the data used for the model. It consists of features and annotations.

​	`all_ratings`: A folder containing all the original annotations and gold-standard on `RECOLA` dataset.

​	`mfb-mean`: Log mel filter bank features extracted from `wav` files of `RECOLA` averaged for each four adjacent features.

​	`GS_GenS-1.8-arousal`: A folder containing the generated gold-standard for arousal mentioned on the paper.

​	`GS_GenS-1.8-valence`: A folder containing the generated gold-standard for valence mentioned on the paper.
