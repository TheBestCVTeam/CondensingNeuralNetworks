# NeuralNetworkInitial
A simple neural net with pytorch and hopeful using resnet.

# Contributing Authors
- Sai Tedla
- Olga Klushina
- Henilkumar Patel
- Sai Varun Ramavath
- Andres Rojas
- Chester Wyke
- Beixuan Yang

# How to run code locally

To run the code simply run main_.py in the root of the src folder.
- Navigate to the src folder and execute: ```python main.py``` 

Below is a brief description of what code is each of the folders.

# Packages needed
Note conda is not strictly required but example code uses it. Feel free to get 
via an alternative mechanism. Commands should be used in the correct environment
the one way it to click on "Terminal" at the bottom of PyCharm.
- [PIL](https://anaconda.org/anaconda/pillow)
    - ```conda install -c anaconda pillow```
- [pytorch](https://anaconda.org/pytorch/pytorch)
    - ```conda install -c pytorch pytorch```
- [torchvision](https://anaconda.org/pytorch/torchvision)
    - ```conda install -c pytorch torchvision```
- [opencv](https://anaconda.org/conda-forge/opencv)
    - ```conda install -c conda-forge opencv```
- [scikit-learn](https://anaconda.org/anaconda/scikit-learn)
    - ```conda install -c anaconda scikit-learn ```
- [matplotlib](https://anaconda.org/conda-forge/matplotlib)
    - ```conda install -c conda-forge matplotlib```     

# Folder Structure Brief explanation
- [Data Bundling](#data-bundling)
- [Dataset](#dataset)
- [Dataset Info](#dataset-info)
- [Eval](#eval)
- [Filter](#filter)
- [Local](#local)
- [Utils](#utils)
  

## Data Bundling
Code for creating data bundles from the full dataset to use for training and 
testing

## Dataset
Code for managing the access and handling of the dataset during training and 
testing

## Dataset Info
Stores scripts that return information about the dataset

## Eval
Houses code related to evaluation of the models

## Filter
Base class and subclasses for filters

## Local
Settings specific to a particular host.

There are template files that show what settings need to be provided. You can 
just copy the template files and change the prefix of the file from "template" 
to "loc". Then change any setting that do not match what is needed for your 
computer.


## Utils
### config.py
    Central file to store all configuration/settings code.
    To prevent import problems this file should not depend on any of the other
    files in the project. Therefore where classes need to be referenced they
    should be stored in this class as strings and converted at runtime to
    class type variables.
### misc_func.py
    Stores functions that do not depend on any other modules. Allows import
    without concern for import cycles.
