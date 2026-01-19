# TYrPPG (Mambaout-based rPPG)

 [IEEE WI-IAT 25' AI4SG Workshop Oral] TYrPPG: Uncomplicated and Enhanced Learning Capability rPPG for Remote Heart Rate Estimation

You can find our paper [here](https://arxiv.org/abs/2511.05833).

<img src="https://github.com/Taixi-CHEN/TYrPPG/blob/main/Loss.png" width="700">

### This is the first Mambaout-based (Gated CNN) rPPG algorithm, with the proposed video-MMD loss for optimization

---

## SetUp


**STEP1: bash setup.sh**

**STEP2: conda activate rppg**

**STEP3: pip install -r requirements.txt**

---

## Experiment


You need to change the data path and run the code using the ```python main.py --config_file ./configs/XX.yaml```,  XX should be the yaml file inside the configs.

---

## Evaluation and Training
### Testing

STEP1: Download the needed datasets (MMPD, PURE, or others)

STEP2: Change the ```./configs/PURE_TYrPPG.yaml```, **ensure the TOOLBOX_MODE is "only_test"**

STEP3: Run model using ```python main.py --config_file ./configs/PURE_TYrPPG.yaml```

### Training

STEP1: Download the needed datasets (MMPD, PURE, or others)

STEP2: Change the ```./configs/PURE_TYrPPG.yaml```, **ensure the TOOLBOX_MODE is "train_and_test"**, change the train data path and test data path to do anything you want

STEP3: Run model using ```python main.py --config_file ./configs/PURE_TYrPPG.yaml```
























