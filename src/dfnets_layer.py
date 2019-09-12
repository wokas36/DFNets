from keras import initializers, activations, constraints, regularizers
import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from dfnets_conv_op import *

class DFNets(Layer):
    
    def __init__(self,
                output_dim,
                arma_conv_AR,
                arma_conv_MA,
                input_signal,
                num_filters=1,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):
        super(DFNets, self).__init__(**kwargs)
        
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.arma_conv_AR = arma_conv_AR
        self.arma_conv_MA = arma_conv_MA
        self.input_signal = input_signal
        
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer.__name__ = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
    def build(self, input_shape):
        
        self.input_dim = input_shape[-1]
        
        if self.num_filters is not None:
            ar_kernel_shape = (self.num_filters * self.input_dim, self.output_dim)
            ma_kernel_shape = (self.num_filters * self.input_signal.shape[1].value, self.output_dim)
        else:
            ar_kernel_shape = (self.input_dim, self.output_dim)
            ma_kernel_shape = (self.input_signal.shape[1].value, self.output_dim)
        
        self.ar_kernel = self.add_weight(shape=ar_kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        self.ma_kernel = self.add_weight(shape=ma_kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        if(self.use_bias):
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.kernel_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.built = True
        
    def call(self, input):
        
        output = dfnets_graph_conv(input,
                                 self.num_filters,
                                 self.arma_conv_AR,
                                 self.arma_conv_MA,
                                 self.input_signal,
                                 self.ar_kernel,
                                 self.ma_kernel)
        if(self.use_bias):
            output = K.bias_add(output, self.bias)
        if(self.activation is not None):
            output = self.activation(output)
        
        return output
    
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim)
        return output_shape
    
    def get_config(self):
        
        config = {
            'output_dim': self.output_dim,
            'num_filters': self.num_filters,
            'arma_conv_AR': self.arma_conv_AR,
            'arma_conv_MA': self.arma_conv_MA,
            'input_signal': self.input_signal,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        
        base_config = super(DFNets, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))