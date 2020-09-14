import pickle
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline



epochs = 50
gaussain_noise_std = 0.01 # change this number for each noise level
pf = open('./saved_models_with_KL_hidden_unit_54/VDP_lstm_epoch_{}/test_results_random_noise_{}/uncertainty_info.pkl'.format(epochs,gaussain_noise_std), 'rb')         
logits_, sigma_, true_x, true_y, test_acc= pickle.load(pf)                                                   
pf.close()


var = np.zeros([400, 25])
a = 0
for i in range(400):
    for j in range(25):
        predicted_out = np.argmax(logits_[i,j,:])
        s = sigma_[i,j, int(predicted_out), int(predicted_out)]
        if(np.abs(a - s)> 10000 ):
            var[i,j] = a
        else:
            var[i,j] = s
        a = var[i,j]
      
print(np.mean(var))
print(test_acc)
