# Hamiltonian Monte Carlo applied to a Deep Network

* Trained on Tensorflow Probability 0.23.0

## Contents
* [Repository Files](#repository-files)
* [Model](#model)
* [Dataset](#dataset)
* [Results](#results)

## Repository Files

* model.py - This file contains the model along with the likelihood function.
* train_and_test.py - This file contains the preprocessing functions and the training and testing scripts.
* tfp_hmc_weights.pkl - This files contains the trained weights for the model.

## Model

The model consists of a sequence of linear layers with a Categorical distribution layer at the end. The priors of the weights and biases are defined as normal distributions. The log probability function is defined as a weighted sum of the log probability of the prior and the log probability of the likelihood function. More weight is placed on the log probability of the likelihood function.

This model was trained using Hamiltonian Monte Carlo sampling. 

## Dataset

The model was trained on the Heart Disease dataset. This dataset contains information on 303 patients and this information is used to predict whether a patient has heart disease. 

Reference to the dataset:

```Bibtex
@misc{misc_heart_disease_45,
  author       = {Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert},
  title        = {{Heart Disease}},
  year         = {1988},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C52P4X}
}
```

## Results

The model was able to achieve an 80% accuracy on the test dataset.<br>
The dataset used here was quite small. A higher accuracy level could be achieved by:

* using a larger dataset

* using more burnin steps during HMC sampling

* taking a larger number of samples during HMC sampling

