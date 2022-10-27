This is the source code for reproducing the attack presented in "One Truth Prevails" paper.
More precisely, this repository is designed to reproduce the results of Section 4 of the paper, that is, the evaluation of the estimation method of private keys and the partial key exposure attack.

# Quick Start Guide

## DL-SCA

1. Clone this repository to get the source code for the experiment.

    ```git clone https://github.com/ECSIS-lab/one_truth_prevails.git```

2. Download the zip archive of datasets from https://drive.google.com/drive/folders/1o6eWsfopu70qAVZZx42V4zeM0clA-CL3?usp=sharing and unzip it.

3. Place the unzipped datasets referring to the following "Repository structure".

4. Install the modules for the use of our source code dl.py

    ```pip install numpy tensorflow pandas tqdm```

5. Let training and test datasets of profiring traces be p.npy and a.npy, respectively, and put them at same directory with dl.py.

6. Execute dl.py.

   ```python dl.py``` 

## Repository structure
The structure of this repository is shown below:
```
.
├── 1024
│    ├── p.npy                          ┐
│    ├── p_labels                       ┼─ Download from Google Drive
│    ├── a.npy                          ┘
│    ├── a_labels.npy                   ┐
│    ├── key_for_partial_key_exposure   ┼─ Download from Github
│    ├── partial_key_exposure.py        ┤
│    └── dl.py                          ┘
└── 2048
     ├── p.npy                          ┐
     ├── p_labels                       ┼─ Download from Google Drive
     ├── a.npy                          ┘
     ├── a_labels.txt                   ┐
     ├── key_for_partial_key_exposure   ┼─ Download from Github
     ├── partial_key_exposure.py        ┤
     └── dl.py                          ┘
```

### How to interpret the training result

After executing dl.py, you can find the NN loss and accuracy for training, validation.
You may observe the training(i.e. profiling) accuracy as "acc", and validation(i.e. attack) accuracy as "val_acc".
As we use attack traces for target keys for validation traces, we can consider val_acc indicates how efficiently our NN distinguishes true or dummy load during processing of private keys under the attack. (Note that validation traces NOT affect the training of NN.)

## Partial Key Exposure Attack

1. Let the test case key list of the partial key exposure attack be key_for_partial_key_exposure.txt, put them at same directory with partial_key_exposure.py.

2. Edit values named "num_error" and "key_index" under ```if __name__ == "__main__":```

3. Execute partial_key_exposure.py.

   ```python partial_key_exposure.py``` 

### How to interpret the training result

When the process ends, you may obtain files named "result_time.csv" and "result_branch.csv".
In each files, the elapsed time of whole algorithm execution and the length of queue at end of the process are recorded, respectively.
In both of these csv tables, the row index corresponds to the key index, and column index corresponds to the number of errors in the key.
