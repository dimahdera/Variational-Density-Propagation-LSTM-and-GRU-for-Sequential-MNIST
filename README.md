# Variational-Density-Propagation-LSTM-for-Sequential-MNIST
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

To compute the average output variance of the test set (after removing outliers), we follow:

import pickle\
import numpy as np\
import matplotlib.pyplot as plt\
%matplotlib inline

epochs = 50\
gaussain_noise_std = 0.01\
pf = open('./saved_models_with_KL_hidden_unit_54/VDP_lstm_epoch_{}/test_results_random_noise_{}/uncertainty_info.pkl'.format(epochs,gaussain_noise_std), 'rb')        
logits_, sigma_, true_x, true_y, test_acc= pickle.load(pf)                                                  
pf.close()

 
var = np.zeros([400, 25])\
a = 0\
for i in range(400):\
....for j in range(25):\
........predicted_out = np.argmax(logits_[i,j,:])\
........s = sigma_[i,j, int(predicted_out), int(predicted_out)]\
........if(np.abs(a - s)> 10000 ):\
............var[i,j] = a\
     .  .else:\
            var[i,j] = s\
        a = var[i,j]\
      
print(np.mean(var))\
print(test_acc)

