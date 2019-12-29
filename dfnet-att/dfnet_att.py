import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# coding: utf-8

# In[1]:


from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, ReLU, LeakyReLU, concatenate, Input
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.constraints import unit_norm
from keras import optimizers
from keras import regularizers
from keras import initializers
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from scipy.linalg import fractional_matrix_power
import tensorflow as tf
import numpy as np

from utils import *
from dfnets_optimizer import *
from dfnets_layer import DFNetsATT

import warnings
warnings.filterwarnings('ignore')


# In[2]:
# In[3]:


X, A, S, Y = load_data_attention(dataset='cora')
A = np.array(A.todense())


# In[4]:


_, Y_val, _, train_idx, val_idx, test_idx, train_mask = get_splits(Y, S)
train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)
labels = np.argmax(Y, axis=1) + 1

# Normalize X
#X /= X.sum(1).reshape(-1, 1)
X = np.array(X)

Y_train = np.zeros(Y.shape)
labels_train = np.zeros(labels.shape)
Y_train[train_idx] = Y[train_idx]
labels_train[train_idx] = labels[train_idx]

Y_test = np.zeros(Y.shape)
labels_test = np.zeros(labels.shape)
Y_test[test_idx] = Y[test_idx]
labels_test[test_idx] = labels[test_idx]


# In[5]:


#Identity matrix for self loop.
I = np.matrix(np.eye(A.shape[0]))
A_hat = A + I

#Degree matrix.
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

#Laplacian matrix.
L = I - (fractional_matrix_power(D_hat, -0.5) * A_hat * fractional_matrix_power(D_hat, -0.5))
L = L - ((lmax(L)/2) * I)

lambda_cut = 0.5

def step(x, a):
    for index in range(len(x)):
        if(x[index] >= a):
            x[index] = float(1)
        else:
            x[index] = float(0)
    return x
    
response = lambda x: step(x, lmax(L)/2 - lambda_cut)

# Since the eigenvalues might change, sample eigenvalue domain uniformly
mu = np.linspace(0, lmax(L), 100)

#AR filter order.
Ka = 5

#MA filter order.
Kb = 3

#The parameter 'radius' controls the tradeoff between convergence efficiency and approximation accuracy. 
#A higher value of 'radius' can lead to slower convergence but better accuracy.
radius = 0.90

b, a, rARMA, error = dfnets_coefficients_optimizer(mu, response, Kb, Ka, radius)


# In[6]:


h_zero = np.zeros(L.shape[0])

def L_mult_numerator(coef):
    y = coef.item(0) * np.linalg.matrix_power(L, 0)
    for i in range(1, len(coef)):
        x = np.linalg.matrix_power(L, i)
        y = y + coef.item(i) * x

    return y

def L_mult_denominator(coef):
    y_d = h_zero
    for i in range(0, len(coef)):
        x_d = np.linalg.matrix_power(L, i+1)
        y_d = y_d + coef.item(i) * x_d
    
    return y_d

poly_num = L_mult_numerator(b)
poly_denom = L_mult_denominator(a)

arma_conv_AR = K.constant(poly_denom)
arma_conv_MA = K.constant(poly_num)


# In[7]:


def dense_factor(inputs, input_signal, num_nodes, droput):
    
    h_1 = BatchNormalization()(inputs)
    h_1 = DFNetsATT(num_nodes, 
			  A_hat, 
			  arma_conv_AR, 
			  arma_conv_MA, 
			  input_signal,
			  num_attention_heads=8, 
			  attention_combine='concat', 
			  attention_dropout=0.5,
			  kernel_initializer=initializers.glorot_normal(seed=1), 
			  kernel_regularizer=l2(9e-4), 
			  kernel_constraint=unit_norm(),
			  use_bias=True,
			  bias_initializer=initializers.glorot_normal(seed=1), 
			  bias_constraint=unit_norm())(h_1)
    h_1 = ReLU()(h_1)
    output = Dropout(droput)(h_1)
    return output
#3
def dense_block(inputs):

    concatenated_inputs = inputs
    
    num_nodes = [8, 16, 32, 64]
    droput = [0.9, 0.9, 0.9, 0.9]

    for i in range(4):
        x = dense_factor(concatenated_inputs, inputs, num_nodes[i], droput[i])
        concatenated_inputs = concatenate([concatenated_inputs, x], axis=1)

    return concatenated_inputs


# In[8]:


def dense_block_model(x_train):
    
    inputs = Input((x_train.shape[1],))

    x = dense_block(inputs)
    
    predictions = Dense(7, kernel_initializer=initializers.glorot_normal(seed=1), 
                        kernel_regularizer=regularizers.l2(1e-10), 
                        kernel_constraint=unit_norm(), 
                        activity_regularizer=regularizers.l2(1e-10), 
                        use_bias=True, 
                        bias_initializer=initializers.glorot_normal(seed=1), 
                        bias_constraint=unit_norm(), 
                        activation='softmax', name='fc_'+str(1))(x)

    model = Model(input=inputs, output=predictions)
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002), metrics=['acc'])
    
    return model

model = dense_block_model(X)
model.summary()


# In[9]:


nb_epochs = 200

class_weight = class_weight.compute_class_weight('balanced', np.unique(labels_train), labels_train)
class_weight_dic = dict(enumerate(class_weight))

for epoch in range(nb_epochs):
    model.fit(X, Y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, 
              class_weight=class_weight_dic, verbose=0)
    Y_pred = model.predict(X, batch_size=A.shape[0])
    _, train_acc = evaluate_preds(Y_pred, [Y_train], [train_idx])
    _, test_acc = evaluate_preds(Y_pred, [Y_test], [test_idx])
    print("Epoch: {:04d}".format(epoch), "train_acc= {:.4f}".format(train_acc[0]), "test_acc= {:.4f}".format(test_acc[0]))
