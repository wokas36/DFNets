import keras.backend as K
import tensorflow as tf
import numpy as np

def dfnets_graph_conv(x, num_filters, arma_conv_AR, arma_conv_MA, input_signal, ar_kernel, ma_kernel):
        
        y1 = K.dot((-arma_conv_AR), x)
        y2 = K.dot(arma_conv_MA, input_signal)
        
        conv_op_y1 = tf.split(y1, num_filters, axis=0)
        conv_op_y1 = K.concatenate(conv_op_y1, axis=1)
        conv_op_y1 = K.dot(conv_op_y1, ar_kernel)
        
        conv_op_y2 = tf.split(y2, num_filters, axis=0)
        conv_op_y2 = K.concatenate(conv_op_y2, axis=1)
        conv_op_y2 = K.dot(conv_op_y2, ma_kernel)
        
        conv_out = conv_op_y1 + conv_op_y2
        
        return conv_out