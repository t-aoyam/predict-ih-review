# Pr


This is a repo for a paper under submission "Predicting the Emergence of Induction Heads in Language Model Pretraining"

## Setting up the Environment

```
$ conda create -n predict-ih python=3.8
$ conda activate predict-ih
$ pip install -r requirements.txt
```

## Repo Structure

```
predict-ih-review/
│  README.md
│  .gitignore
│
├─ src/                  
│   ├─ data/             # lightweight data-loading helpers
│   │    ├─ configs/     # .json configs for training
│   │    └─ .../         # all other result files
│   │
│   ├─ models/           # model classes
│   ├─ training/         # training loops, trainer classes, etc.
│   ├─ evaluation/       # metrics, analysis utilities
│   └─ utils/
│
├─ notebooks/
│   └─ figures.ipynb     # all figures in the paper can be generated here
│
├─ scripts/              # .sh scripts
│   ├─ train.sh
│   └─ evaluate.sh
│
├─ models/               # all pytorch models
│   └─ model-name
│        └─ checkpoint-xxx/
│
└─ data/                 # data for pretraining, evaluation, human eye-tracking/self-paced reading data, etc.
    ├─ README.md         # where to download, expected hashes, etc.
    ├─ configs/     # .json configs for training
    └─ .../         # all other result files
```
## How to Run the Code

### Training the Model

See `scripts/training_sample.sh` for how to train a model.

### Evaluating the Model

See `scripts/evaluation_sample.sh` for how to evaluate a trained LM on prefix-matching score, logit attribution, associative recall (accuracy), and associative recall (mean rank).
