# About

Here you can find accompanying information regarding the methods proposed in the manuscript: "Stratified neural networks in a time-to-event setting". 

Specifically 

1. an implementation of [stratified loss functions](src/modules/torch_models.py) for training deep neural network on time to event (survival) data

2. Python scripts for reproducing the analyses shown in the manuscript ([training of networks](src/train_models.py) and subsequent [evaluation](src/evaluate_models.py) as well as the [transfer learning training](src/train_models_transfer.py) approach together with corresponding [transfer learning evaluation](src/evaluate_models_transfer.py).

3. An exemplary [Jupyter Notebook](notebook.ipynb) which, based on a small data example, demonstrates how the proposed methods can be applied.


## Installation
The software is written in Python and builds on [PyTorch](https://pytorch.org) for training the deep networks, [scikit-learn](https://scikit-learn.org/stable/) and [scikit-survival](https://github.com/sebp/scikit-survival). 

For all our experiments we used Python Version 3.8.3

Create a virtual environment
```
python3 -m venv .venv
```

Activate the virtual environment
```
source .venv/bin/activate
```

Install dependencies (environment needs to be activated)
```
pip install -r requirements.txt
```

Navigate to the folder with the source code
```
cd src
```

In order to train the models you just need to start the corresponding script and parse the tumor type combination you want to analyse as command line arguments. To reproduce the results from the manuscript this would be: 
```
python3 train_models.py BRCA GBM LGG KIRC KICH KIRP
```

You can play around with the hyperparameters provided
in the [config file](optimization_configs.json). If *SAVE=True*, the summary statistics needed for later evaluation are automatically saved into files.

Again, in order to evaluate the fitted models you just need to parse the tumor type combination as command line arguments. For our main results this would be:
```
python3 evaluate_models.py BRCA GBM LGG KIRC KICH KIRP
```

Same procedure can be applied for the transfer learning scripts, i.e. for training use:

```
python3 train_models_transfer.py GBM KIRC
```

And for subsequent evaluation type:
```
python3 train_models_transfer.py GBM KIRC
```
