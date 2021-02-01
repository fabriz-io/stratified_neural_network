# About

Here you can find accompanying information regarding the methods proposed in the manuscript: "Stratified neural networks in a time-to-event setting". 

Specifically 

1. an implementation of [stratified loss functions](src/modules/torch_models.py) for training deep neural network on time to event (survival) data

2. Python scripts for reproducing the analyses shown in the manuscript ([training of networks](src/train_models.py) and subsequent [evaluation](src/evaluate_models.py))

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

In order to reproduce our results, you need 
to create the full data set.
The provided script downloads and reproduces all
preprocessing steps

```
python3 create_data.py
```

Train the models. You can play around with the hyperparameters provided
at the beginning of the script. If "SAVE=True", the summary statistics are saved automatically into files in order to be evaluated later.
```
python3 train_models.py
```

Evaluate the models with the summary files produced during training. 
Note that not all plots are saved automatically into files. We used an interactive environment from which we exported most of the plots.
```
python3 evaluate_models.py
```