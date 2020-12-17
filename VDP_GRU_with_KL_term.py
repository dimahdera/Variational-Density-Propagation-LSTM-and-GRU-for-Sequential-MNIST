import tensorflow as tf
from tensorflow import keras
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit

plt.ioff()
mnist = tf.keras.datasets.mnist

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

# This function performs a multiplication x \Sigma_w x^T in the batch context, where x is a vector and \Sigma_w is a matrix    
def x_Sigma_w_x_T(x, W_Sigma):
  batch_sz = x.shape[0]
  xx_t = tf.reduce_sum(tf.multiply(x, x),axis=1, keepdims=True)               
  xx_t_e = tf.expand_dims(xx_t,axis=2)                                      
  return tf.multiply(xx_t_e, W_Sigma)
  
# This function performs a multiplication w^T \Sigma_i w in the batch context, where w is a vector and \Sigma_i is a matrix
def w_t_Sigma_i_w (w_mu, in_Sigma):
  Sigma_1_1 = tf.matmul(tf.transpose(w_mu), in_Sigma)
  Sigma_1_2 = tf.matmul(Sigma_1_1, w_mu)
  return Sigma_1_2
# This function computes the trace, tr(\Sigma_w \Sigma_in) in the batch context, where \Sigma_w and \Sigma_in are two matrices
def tr_Sigma_w_Sigma_in (in_Sigma, W_Sigma):
  Sigma_3_1 = tf.linalg.trace(in_Sigma)
  Sigma_3_2 = tf.expand_dims(Sigma_3_1, axis=1)
  Sigma_3_3 = tf.expand_dims(Sigma_3_2, axis=1)
  return tf.multiply(Sigma_3_3, W_Sigma)

# This function propagation sigma through the activation function after cmputing the gradient 
def activation_Sigma (gradi, Sigma_in):
  grad1 = tf.expand_dims(gradi,axis=2)
  grad2 = tf.expand_dims(gradi,axis=1)
  return tf.multiply(Sigma_in, tf.matmul(grad1, grad2))

# This function computes the Hadamard product between two random vectors in the batch context
def Hadamard_sigma(sigma1, sigma2, mu1, mu2):
  sigma_1 = tf.multiply(sigma1, sigma2)
  sigma_2 = tf.matmul(tf.matmul(tf.linalg.diag(mu1) ,   sigma2),   tf.linalg.diag(mu1))
  sigma_3 = tf.matmul(tf.matmul(tf.linalg.diag(mu2) ,   sigma1),   tf.linalg.diag(mu2))
  return sigma_1 + sigma_2 + sigma_3

# This function computes the gradient of the sigmoid function 
def grad_sigmoid(mu_in):
  with tf.GradientTape() as g:
    g.watch(mu_in)
    out = tf.sigmoid(mu_in)
  gradi = g.gradient(out, mu_in) 
  return gradi

# This function computes the gradient of the tanh function
def grad_tanh(mu_in):
  with tf.GradientTape() as g:
    g.watch(mu_in)
    out = tf.tanh(mu_in)
  gradi = g.gradient(out, mu_in) 
  return gradi

# This function performs the multiplication \mu \mu^T in the batch context, where \mu is a vector
def mu_muT(mu1, mu2):
  mu11 = tf.expand_dims(mu1,axis=2)
  mu22 = tf.expand_dims(mu2,axis=1)
  return tf.matmul(mu11, mu22)

# This function computes the sigma regulirizer for the input-recurent weights in the LSTM layer 
def sigma_regularizer1(x):
    input_size = 25.   
    f_s = tf.math.softplus(x) #tf.math.log(1. + tf.math.exp(x)) 
    return - input_size * tf.reduce_mean(1. + tf.math.log(f_s) - f_s , axis=-1)
# This function computes the sigma regulirizer for the recurent-recurent weights in the LSTM layer and sigma regulirizer of the FC layer
def sigma_regularizer2(x):      
    f_s = tf.math.softplus(x)#  tf.math.log(1. + tf.math.exp(x))
    return - tf.reduce_mean(1. + tf.math.log(f_s) - f_s , axis=-1)
# The class that propagates the mean and covariance matrix of the variational distribution through the GRU cell        
class densityPropGRUCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([units, units])]
        super(densityPropGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        # We define the initial conditions of the mean and covariance metrices of all gates in the GRU cell. We also add the regulirization terms 
        input_size = input_shape[-1]
        ini_sigma = -2.2
        min_sigma = -4.5
        init_mu = 0.05      
        seed_ = None
        tau1 = 1.
        tau2 = 100./self.units        
        
        self.U_z = self.add_weight(shape=(input_size, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None), regularizer=tf.keras.regularizers.l2(tau1),name='U_z', trainable=True)
        self.uz_sigma = self.add_weight(shape=(self.units,),  initializer=tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None),regularizer=sigma_regularizer1, name='uz_sigma', trainable=True)       

        self.W_z = self.add_weight(shape=(self.units, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None), regularizer=tf.keras.regularizers.l2(tau2), name='W_z', trainable=True)
        self.wz_sigma = self.add_weight(shape=(self.units,),  initializer=tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None),regularizer=sigma_regularizer2, name='wz_sigma', trainable=True)

        self.U_r = self.add_weight(shape=(input_size, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None), regularizer=tf.keras.regularizers.l2(tau1), name='U_r',trainable=True)
        self.ur_sigma = self.add_weight(shape=(self.units,),  initializer=tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None),regularizer=sigma_regularizer1, name='ur_sigma', trainable=True)

        self.W_r = self.add_weight(shape=(self.units, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None),regularizer=tf.keras.regularizers.l2(tau2),  name='W_r', trainable=True)
        self.wr_sigma = self.add_weight(shape=(self.units,),  initializer=tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None),regularizer=sigma_regularizer2, name='wr_sigma', trainable=True)

        self.U_h = self.add_weight(shape=(input_size, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None),regularizer=tf.keras.regularizers.l2(tau1),  name='U_h',trainable=True)
        self.uh_sigma = self.add_weight(shape=(self.units,),  initializer=tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None),regularizer=sigma_regularizer1, name='uh_sigma', trainable=True)

        self.W_h = self.add_weight(shape=(self.units, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None),regularizer=tf.keras.regularizers.l2(tau2),  name='W_h', trainable=True)
        self.wh_sigma = self.add_weight(shape=(self.units,),  initializer=tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None),regularizer=sigma_regularizer2, name='wh_sigma', trainable=True)
        
        self.built = True
    def call(self, inputs, states):        
        # state should be in [(batch, units), (batch, units, units)], mean vecor and covaraince matrix
        prev_state, Sigma_state = states        
        ## Update Gate - propagate the mean and covariance matrix through the update gate
        z = tf.sigmoid(tf.matmul(inputs, self.U_z) + tf.matmul(prev_state, self.W_z))
        Uz_Sigma = tf.linalg.diag(tf.math.softplus(self.uz_sigma)  )                                          
        Sigma_Uz = x_Sigma_w_x_T(inputs, Uz_Sigma)         
        ################
        Wz_Sigma = tf.linalg.diag(tf.math.softplus(self.wz_sigma)  )               
        Sigma_z1 = w_t_Sigma_i_w (self.W_z, Sigma_state)
        Sigma_z2 = x_Sigma_w_x_T(prev_state, Wz_Sigma)                                  
        Sigma_z3 = tr_Sigma_w_Sigma_in (Sigma_state, Wz_Sigma)
        Sigma_out_zz = Sigma_z1 + Sigma_z2 + Sigma_z3 + Sigma_Uz
        ################
        gradi_z = grad_sigmoid(tf.matmul(inputs, self.U_z) + tf.matmul(prev_state, self.W_z))
        Sigma_out_z = activation_Sigma(gradi_z, Sigma_out_zz)
        ###################################
        ###################################
        ## Reset Gate - propagate the mean and covariance matrix through the reset gate
        r = tf.sigmoid (tf.matmul(inputs, self.U_r) + tf.matmul(prev_state, self.W_r))
        Ur_Sigma = tf.linalg.diag(tf.math.softplus(self.ur_sigma) )                                     
        Sigma_Ur = x_Sigma_w_x_T(inputs, Ur_Sigma)
        ################
        Wr_Sigma = tf.linalg.diag(tf.math.softplus(self.wr_sigma)   )              
        Sigma_r1 = w_t_Sigma_i_w(self.W_r, Sigma_state)
        Sigma_r2 = x_Sigma_w_x_T(prev_state, Wr_Sigma)                                  
        Sigma_r3 = tr_Sigma_w_Sigma_in (Sigma_state, Wr_Sigma)
        Sigma_out_rr = Sigma_r1 + Sigma_r2 + Sigma_r3 + Sigma_Ur
        ################        
        gradi_r = grad_sigmoid(tf.matmul(inputs, self.U_r) + tf.matmul(prev_state, self.W_r))
        Sigma_out_r = activation_Sigma(gradi_r, Sigma_out_rr)
        ###################################
        ###################################
        ## Intermediate Activation - propagate the mean and covariance matrix through the intermidiate activation
        h = tf.tanh(tf.matmul(inputs, self.U_h) + tf.matmul(tf.multiply(prev_state, r), self.W_h))
        Uh_Sigma = tf.linalg.diag(tf.math.softplus(self.uh_sigma) )                                   
        Sigma_Uh = x_Sigma_w_x_T(inputs, Uh_Sigma)
        ################
        sigma_g = Hadamard_sigma(Sigma_state, Sigma_out_r, prev_state, r)
        ################
        Wh_Sigma = tf.linalg.diag(tf.math.softplus(self.wh_sigma)    )              
        Sigma_h1 = w_t_Sigma_i_w (self.W_h, sigma_g)
        Sigma_h2 = x_Sigma_w_x_T(tf.multiply(prev_state, r), Wh_Sigma)                                   
        Sigma_h3 = tr_Sigma_w_Sigma_in (sigma_g, Wh_Sigma)
        Sigma_out_hh = Sigma_h1 + Sigma_h2 + Sigma_h3 + Sigma_Uh
        ################
        gradi_h = grad_tanh(tf.matmul(inputs, self.U_h) + tf.matmul(tf.multiply(prev_state, r), self.W_h))
        Sigma_out_h = activation_Sigma(gradi_h, Sigma_out_hh)
        ###################################
        ###################################
        ## Current State   - propagate the mean and covariance matrix through the current state        
        mu_out = tf.multiply(z, prev_state) + tf.multiply(1-z, h)
        sigma_a = Hadamard_sigma(Sigma_out_z, Sigma_state,z, prev_state)
        sigma_b = Hadamard_sigma(Sigma_out_z, Sigma_out_h, 1-z, h)
        mu_s_muhT = mu_muT(prev_state, h)
        mu_h_mu_sT = mu_muT(h, prev_state)
        sigma_ab = tf.multiply(Sigma_out_z, mu_s_muhT)
        sigma_abT = tf.multiply(Sigma_out_z, mu_h_mu_sT)
        Sigma_out = sigma_a + sigma_b - sigma_ab - sigma_abT
        
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out))) 
        
        output = mu_out
        new_state = (mu_out, Sigma_out)
        return output, new_state


# The Linear class that propagates the mean and covariance matrix of the variational distribution through the fully-connected layer	
# Linear Class - Second Layer (RV * RV)
class LinearNotFirst(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units):
        super(LinearNotFirst, self).__init__()
        self.units = units

    def build(self, input_shape):
        # We define the initial conditions of the mean and covariance metrices in the fully-connected layer. We also add the regulirization term
        ini_sigma = -2.2
        min_sigma = -4.5
        tau = 100./input_shape[-1]        
        self.w_mu = self.add_weight(name='w_mu', shape=(input_shape[-1], self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=0.05, seed=None),  
                    regularizer=tf.keras.regularizers.l2(tau),  trainable=True, )
        self.w_sigma = self.add_weight(name='w_sigma',
            shape=(self.units,),
            initializer=tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None), regularizer=sigma_regularizer2,
            trainable=True,        )
    def call(self, mu_in, Sigma_in):
        # Propagate the mean and covariance matrix through the fully-connected layer
        mu_out = tf.matmul(mu_in, self.w_mu)      
        W_Sigma = tf.linalg.diag(tf.math.softplus(self.w_sigma)  )
               
        Sigma_1 = w_t_Sigma_i_w (self.w_mu, Sigma_in)
        Sigma_2 = x_Sigma_w_x_T(mu_in, W_Sigma)                                 
        Sigma_3 = tr_Sigma_w_Sigma_in (Sigma_in, W_Sigma)
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3       
        
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)  
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))      
        return mu_out, Sigma_out


# The class that propagates the mean and covariance matrix of the variational distribution through the Softmax layer	
class mysoftmax(keras.layers.Layer):
    """Mysoftmax"""

    def __init__(self):
        super(mysoftmax, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.softmax(mu_in)
        pp1 = tf.expand_dims(mu_out, axis=2)
        pp2 = tf.expand_dims(mu_out, axis=1)
        ppT = tf.matmul(pp1, pp2)
        p_diag = tf.linalg.diag(mu_out)
        grad = p_diag - ppT
        Sigma_out = tf.matmul(grad, tf.matmul(Sigma_in, tf.transpose(grad, perm=[0, 2, 1])))
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)        
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))
        return mu_out, Sigma_out


# the log-likelihood of the objective function
# Inputs: 
#       y_pred_mean: The output mean vector (predictive mean).
#       y_pred_sd: The output covariance matrix (predictive covariance matrix).
#       y_test: The ground truth prediction vector
#       num_labels: the number of classes in the dataset. It is 10 for the MNIST dataset
#       batch_size: the size of the mini-batches
# Output:
#       the expected log-likelihood term of the objective function  
def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels, batch_size):    
    y_pred_sd_ns = y_pred_sd 
    s, u, v = tf.linalg.svd(y_pred_sd_ns, full_matrices=True, compute_uv=True)	
    s_ = s + 1.0e-4
    s_inv = tf.linalg.diag(tf.math.divide_no_nan(1., s_) )    
    y_pred_sd_inv = tf.matmul(tf.matmul(v, s_inv), u, transpose_b=True) 
    mu_ =  tf.expand_dims(y_pred_mean - y_test, axis=1) 
    mu_sigma = tf.matmul(mu_ ,  y_pred_sd_inv)     
    loss1 =  tf.squeeze(tf.matmul(mu_sigma , mu_, transpose_b=True))
        
    if tf.reduce_any(tf.math.greater(s , tf.constant(1e-10, shape=[batch_size, num_labels]))):
        loss1 = tf.math.add(loss1, tf.math.reduce_sum(tf.math.log(s_), axis =-1) )
                
    loss = tf.reduce_mean(loss1)
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
    return (loss), tf.reduce_mean(tf.squeeze(tf.matmul(mu_sigma , mu_, transpose_b=True))), tf.reduce_mean(tf.math.reduce_sum(tf.math.log(s_), axis =-1))    

# This class defines the netwrok by calling all layers     
class Density_prop_GRU(tf.keras.Model):
  def __init__(self, units, name=None):
    super(Density_prop_GRU, self).__init__()
    self.units = units
    self.cell = densityPropGRUCell(self.units)
    self.rnn = tf.keras.layers.RNN(self.cell, return_state=True) 
    self.linear_1 = LinearNotFirst(10)
    self.mysoftma = mysoftmax()
  def call(self, inputs, training=True):
    xx = self.rnn(inputs)
    x, mu_state, sigma_state = xx
    m, s = self.linear_1(x, sigma_state)
    outputs, Sigma = self.mysoftma(m, s)  
    Sigma = tf.where(tf.math.is_nan(Sigma), tf.zeros_like(Sigma), Sigma)
    Sigma = tf.where(tf.math.is_inf(Sigma), tf.zeros_like(Sigma), Sigma)       
    return outputs, Sigma

#The main function: 
# Inputs: 
#      time_step: the time step in the GRU layer (28).
#      input_dim:the dimensionality of the input vector (25).
#      units: The number of hidden units.
#      number_of_classes: the number of classes. It is 10 for the MNIST dataset (10 classes).
#      batch_size: the batch size. It is hyper-parameters. We choose it to be 50. 
#      epochs: the number of epochs. 
#      lr : the learning rate of the trainting.
#      kl_factor: the KL weighting factor
#      Random_noise: decide if we want to apply Gaussian noise or not (True or False).
#      gaussain_noise_std: the standard deviation of the Gaussian noise. 
#      Training = decide if we want to do training or loading a trained model for testing (True or False).
#      repeat_initial: decide if we want to load previous initail coditions that we used before or not (True or False). 
#      continue_train: decide if we want to continue training or start training from scratch (True or False).
#      saved_model_epochs: the number of epochs of the saved model
# Output:
#       The function saves the model after training, the weights, training, validation and test accuracies and a text file that shows for example the following: 
##                 Input Dimension : 25
##                 No Hidden Nodes : 64
##                 Output Size : 10
##                 No of epochs : 40
##                 Learning rate : 0.001
##                 time step : 28
##                 batch size : 20
##                 KL term factor : 0.001
##                ---------------------------------
##                 Total run time in sec : 59631.23083720356
##                 Averaged Training  Accuracy : 0.9779247045516968
##                 Averaged Validation Accuracy :  0.9762003421783447
##                 Averaged Training  error : -42.55049514770508
##                 Averaged Validation error : -35.990848541259766
##                ---------------------------------
##                ---------------------------------
#       The main function also plot the training and validation accuracies vs. number of epochs.
#
def main_function(time_step = 28, input_dim = 25, units = 64, number_of_classes = 10 , batch_size = 20, epochs = 40, lr = 0.001, kl_factor=0.001,
        Random_noise=False, gaussain_noise_std=0, Training=True, continue_training = False, saved_model_epochs=50):    

    PATH = './saved_models/VDP_gru_epoch_{}/'.format(epochs)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0 
    one_hot_y_train = tf.one_hot(y_train.astype(np.float32), depth=10)
    one_hot_y_test = tf.one_hot(y_test.astype(np.float32), depth=10)

    # Just to make lenght width of the image of different size
    # Each MNIST image batch is a tensor of shape (batch_size, 28, 25).
    # Each input sequence will be of size (28, 25) (height is treated like time).
    x_train = x_train[:,:,:25]
    x_test = x_test[:,:,:25]
    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, one_hot_y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)        
    # Cutom Trianing Loop with Graph
    gru_model = Density_prop_GRU(units, name = 'vdp_gru')    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr )#, clipnorm=10.)#, clipvalue=1.)
    
    @tf.function  # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            logits, sigma = gru_model(x)      
            loss_final, loss1, loss2  = nll_gaussian(y, logits,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-10),
                                       clip_value_max=tf.constant(1e+6)), number_of_classes , batch_size)
            
            regularization_loss=tf.math.add_n(gru_model.losses)             
            loss = 0.5 * (loss_final + kl_factor*regularization_loss )            
            gradients = tape.gradient(loss, gru_model.trainable_weights)
            gradients = [(tf.where(tf.math.is_nan(grad), tf.zeros(grad.shape), grad)) for grad in gradients] 
            gradients = [(tf.where(tf.math.is_inf(grad), tf.zeros(grad.shape), grad)) for grad in gradients]
        optimizer.apply_gradients(zip(gradients, gru_model.trainable_weights))        
        return loss, logits,sigma, gradients, regularization_loss, loss1, loss2 
    if Training:
        if continue_training:
            saved_model_path = './saved_models/VDP_gru_epoch_{}/'.format(saved_model_epochs)           
            gru_model.load_weights(saved_model_path + 'vdp_gru_model')
            
        train_acc = np.zeros(epochs) 
        valid_acc = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        start = timeit.default_timer()
        
        for epoch in range(epochs):
          print('Epoch: ', epoch+1, '/' , epochs)    
          acc1 = 0 
          acc_valid1 = 0 
          err1 = 0
          err_valid1 = 0
          tr_no_steps = 0
          va_no_steps = 0
          #--------------------Training----------------
          for step, (x, y) in enumerate(tr_dataset):
              update_progress(step / int(x_train.shape[0] / (batch_size)) )  
              loss, logits ,sigma, gradients, regularization_loss, loss1, loss2  = train_on_batch(x, y)
              err1+= loss    
              corr = tf.equal(tf.math.argmax(logits, axis=1),tf.math.argmax(y,axis=1))
              accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
              acc1+=accuracy
              if step % 50 == 0:                 
                  print("\n Step:", step, "Loss:" , float(err1/(tr_no_steps + 1.)))
                  print("Total Training accuracy so far: %.3f" % float(acc1/(tr_no_steps + 1.)))                                    
              tr_no_steps+=1
          train_acc[epoch] = acc1/tr_no_steps
          train_err[epoch] = err1/tr_no_steps          
          print('Training Acc  ', train_acc[epoch])
          print('Training error  ', train_err[epoch])          
          #-----------------Validation--------------------     
          for step, (x, y) in enumerate(val_dataset):
              update_progress(step / int(x_test.shape[0] / (batch_size)) )               
              logits, sigma = gru_model(x)  
              vloss, loss1, loss2 = nll_gaussian(y, logits,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-10),
                                           clip_value_max=tf.constant(1e+6)), number_of_classes , batch_size)
              regularization_loss=tf.math.add_n(gru_model.losses)
              total_vloss = 0.5 *(vloss + kl_factor*regularization_loss)                             
              err_valid1+= total_vloss
              
              corr = tf.equal(tf.math.argmax(logits, axis=1),tf.math.argmax(y,axis=1))
              accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
              acc_valid1+=accuracy
    
              if step % 100 == 0:
                  print("Step:", step, "Loss:", float(total_vloss))
                  print("Validation accuracy : %.3f" % accuracy)               
              va_no_steps+=1           
          valid_acc[epoch] = acc_valid1/va_no_steps      
          valid_error[epoch] = err_valid1/va_no_steps
          stop = timeit.default_timer()
                    
          gru_model.save_weights(PATH + 'vdp_gru_model')
          print('Total Training Time: ', stop - start)
          print('Training Acc  ', train_acc[epoch])
          print('Validation Acc  ', valid_acc[epoch])
          print('------------------------------------')
          print('Training error  ', train_err[epoch])
          print('Validation error  ', valid_error[epoch])
        #-----------------End Training--------------------------         
        gru_model.save_weights(PATH + 'vdp_gru_model')        
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')
            plt.plot(valid_acc,'r' , label='Validation acc')
            plt.ylim(0, 1.1)
            plt.title("Density Propagation GRU on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'VDP_GRU_on_MNIST_Data_acc.png')
            plt.close(fig)    
    
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training error')
            plt.plot(valid_error,'r' , label='Validation error')
            #plt.ylim(0, 1.1)
            plt.title("Density Propagation GRU on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VDP_GRU_on_MNIST_Data_error.png')
            plt.close(fig)
        f = open(PATH + 'training_validation_acc_error.pkl', 'wb')         
        pickle.dump([train_acc, valid_acc, train_err, valid_error], f)                                                   
        f.close()             
             
        textfile = open(PATH + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Number of classes : ' +str(number_of_classes))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Learning rate : ' +str(lr)) 
        textfile.write('\n time step : ' +str(time_step))
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))         
        textfile.write("\n---------------------------------")         
        if Training: 
            textfile.write('\n Total run time in sec : ' +str(stop - start))
            if(epochs == 1):
                textfile.write("\n Averaged Training  Accuracy : "+ str( train_acc))
                textfile.write("\n Averaged Validation Accuracy : "+ str(valid_acc ))
                
                textfile.write("\n Averaged Training  error : "+ str( train_err))
                textfile.write("\n Averaged Validation error : "+ str(valid_error ))
            else:
                textfile.write("\n Averaged Training  Accuracy : "+ str(np.mean(train_acc[epoch])))
                textfile.write("\n Averaged Validation Accuracy : "+ str(np.mean(valid_acc[epoch])))
                
                textfile.write("\n Averaged Training  error : "+ str(np.mean(train_err[epoch])))
                textfile.write("\n Averaged Validation error : "+ str(np.mean(valid_error[epoch])))
        textfile.write("\n---------------------------------")                
        textfile.write("\n---------------------------------")    
        textfile.close()
    #-------------------------Testing-----------------------------    
    else:
        test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
        gru_model.load_weights(PATH + 'vdp_gru_model')
        test_no_steps = 0        
        acc_test = 0
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, time_step, input_dim])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, number_of_classes])
        logits_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, number_of_classes])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, number_of_classes, number_of_classes])
        for step, (x, y) in enumerate(val_dataset):
          update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
          true_x[test_no_steps, :, :, :] = x
          true_y[test_no_steps, :, :] = y
          if Random_noise:
              noise = tf.random.normal(shape = [batch_size, time_step, input_dim], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
              x = x +  noise         
          logits, sigma = gru_model(x)  
          logits_[test_no_steps,:,:] =logits
          sigma_[test_no_steps, :, :, :]= sigma         
          corr = tf.equal(tf.math.argmax(logits, axis=1),tf.math.argmax(y,axis=1))
          accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
          acc_test+=accuracy

          if step % 50 == 0:              
              print("Total running accuracy so far: %.3f" % accuracy)             
          test_no_steps+=1
       
        test_acc = acc_test/test_no_steps        
        print('Test accuracy : ', test_acc.numpy())        
        
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')         
        pickle.dump([logits_, sigma_, true_x, true_y, test_acc.numpy()], pf)                                                   
        pf.close()
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Number of classes : ' +str(number_of_classes))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Learning rate : ' +str(lr)) 
        textfile.write('\n time step : ' +str(time_step)) 
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))  
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Accuracy : "+ str( test_acc.numpy()))                  
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std ))            
        textfile.write("\n---------------------------------")    
        textfile.close()
            
if __name__ == '__main__':
    main_function()    
