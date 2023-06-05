## Goal-conditioned ISAR


This repo is build on top of  the [Gofar](https://jasonma2016.github.io/GoFAR/). We extend it to support our ISAR algorithm and reproduce more baselines.



## SetUp
### Requirements
- MuJoCo=2.0.0

### Setup Instructions
1. Create conda environment and activate it:
     ```
     conda env create -f environment.yml
     conda activate gofar
     pip install --upgrade numpy
     pip install torch==1.10.0 torchvision==0.11.1 torchaudio===0.10.0 gym==0.17.3
2. Download the offline dataset [here](https://drive.google.com/file/d/1niq6bK262segc7qZh8m5RRaFNygEXoBR/view) and place ```/offline_data``` in the project root directory.

## Experiments
1. The main results can be reproduced by the following command:
```
mpirun -np 3 python train.py --env $ENV --method $METHOD
```
| Flags and Parameters  | Description |
| ------------- | ------------- |
| ``--env $ENV``  | offline GCRL tasks: ```FetchReach, FetchPush, FetchPick, FetchSlide, HandReach```|
| ``--method $METHOD``  | offline GCRL algorithms: ```gofar, gcsl, wgcsl, actionablemodel, ddpg, isar,td3bc,cql,iql, bcq```|

```
mpirun -np 1 python train.py --env $ENV --method $METHOD --relabel False
```
Note that 'gofar' algorithms defaults to not using HER, so this command is only relevant to the baselines. Relevant flags are listed here:
| Flags and Parameters  | Description |
| ------------- | ------------- |
| ``--relabel``  | whether hindsight experience replay is enabled: ``True``, ``False  ``|
| ``--relabel_percent``  | The fraction of minibatch transitions that has relabeled goals: ``0.0, 0.2, 0.5, 1.0``; these are the hyperparameters attempted in the paper, you may try other fractions too.|
| ``--f``  | Choices of f-divergence for GoFAR: ``kl, chi``.
| ``--reward_type``  | Choices of reward function for GoFAR: ``disc, binary``.


## Acknowledgement:
We borrowed some code from the following repositories:
- [Pytorch DDPG-HER Implementation](https://github.com/)
- [Gofar](https://jasonma2016.github.io/GoFAR/)
