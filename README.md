This is a repo for a paper under submission "Predicting the Emergence of Induction Heads in Language Model Pretraining"

## Setting up the Environment

```
$ conda create -n env-name python=3.9
$ conda activate env-name
$ pip install -r requirements.txt
```

## Repo Structure

```
icml-2026-32900/
│  README.md
│  .gitignore
│
├─ src/                  
│   ├─ data/                    # code for data generation
│   │    ├─ generate_data.py    # main data generation code
│   │    └─ .../
│   │
│   ├─ models/                  # model classes
│   ├─ training/                # training loops, trainer classes, etc.
│   ├─ evaluation/              # metrics, analysis utilities
│   └─ utils.py
│
├─ notebooks/
│   └─ figures.ipynb            # all figures in the paper can be generated here
│
├─ scripts/                     # .sh scripts
│   ├─ data.sh
│   ├─ train.sh
│   └─ evaluate.sh
│
├─ models/                      # all pytorch models
│   └─ model-name
│        └─ checkpoint-xxx/
│
└─ data/                        # data for pretraining, evaluation, human eye-tracking/self-paced reading data, etc.
    ├─ configs/                 # .json configs for training
    └─ .../                     # all other result files
```
## How to Run the Code

### Generating the Transition Matrices and Training Data
See `scripts/data.sh` for how to train a model.

### Training the Model

See `scripts/train.sh` for how to train a model.

### Evaluating the Model

See `scripts/eval.sh` for how to evaluate a trained LM on prefix-matching score, logit attribution, associative recall (accuracy), and associative recall (mean rank).
