# ReliableMSTL

This repository contains the code for [Towards more Reliable Transfer Learning](https://arxiv.org/abs/1807.02235) by [Zirui Wang](https://www.cs.cmu.edu/~ziruiw/) and [Jaime Carbonell](https://www.cs.cmu.edu/~jgc/).

Two methods, PW-MSTL & AMSAT, are proposed to tackle the challenge of multi-source transfer learning when sources exhibiting diverse reliabilities.

## Usage

The code has the following structure:

```
ReliableMSTL /
    data/
    MultiSource_[Transfer/Active]_Lerning_Simulator.py
    ReliableMultiSourceModel.py
    config.py
    [Experiment/Probability/Kernel]Util.py
```

The integrated model is implemented in `ReliableMultiSourceModel.py`. The model requires: (1) data for both the source and the target, (2) indices for labeled source samples for each source domain, and (3) hyperparameters settings. Two examples of how to use the model are shown in `MultiSource_[Transfer/Active]_Lerning_Simulator.py`, using dummy data. To see how it works, simply run:

```
python MultiSource_[Transfer/Active]_Lerning_Simulator.py
```



