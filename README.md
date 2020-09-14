# Variational Density Propagation LSTM and GRU for Sequential MNIST
This code is an implementation of the variational density propagation LSTM and GRU frameworks on Sequential MNIST. 
The algorithm is based on the following papers:

[1].	Dimah Dera, Ghulam Rasool, Nidhal C. Bouaynaya, Adam Eichen, Stephen Shanko, Jeff Cammerata and Sanipa Arnold, “Bayes-SAR Net: Robust SAR Image Classiﬁcation with Uncertainty Estimation Using Bayesian Convolutional Neural Network,” IEEE International Radar Conference, Washington, D.C., April 2020.  
[2].	Dimah Dera, Ghulam Rasool and Nidhal C. Bouaynaya, “Extended Variational Inference for Propagating Uncertainty in Convolutional Neural Networks,” IEEE International Workshop on Machine Learning for Signal Processing, October 2019.

This project uses Python 3.6.0. Before running the code, you have to install:

Tensorflow 2.3.0\
Numpy\
Scipy\
Matplotlib\
pickle\
timeit

To compute the average output variance of the test set (after removing outliers), we use:\
Compute_avreage_output_variance.py

