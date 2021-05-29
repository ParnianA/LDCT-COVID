#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%% Libraries
from __future__ import print_function
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras.models import Model
from keras.layers import Conv2D, Input, Reshape, Lambda, BatchNormalization, MaxPooling2D, Multiply,GlobalMaxPooling1D
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.utils import to_categorical
from keras.layers import TimeDistributed


K.set_image_data_format('channels_last')

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)


class Capsule(Layer):
   

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.num_capsule,
        'dim_capsule' : self.dim_capsule,
        'routings':  self.routings,
        'share_weight':self.share_weights,
        
       
           
        })
        return config

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = keras.backend.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    

#load images, labels, and probs from stage one
x_train = np.load('path to images').astype('float32')
y_train=np.load('path to labels')
prob_train=np.load('path to probs')
prob_train=np.expand_dims(prob_train,2)
prob_train=np.repeat(prob_train,3,axis=2)
prob_train[:,:,0]=1-prob_train[:,:,0]
y_train = np.array(to_categorical(y_train))
rand = np.arange(len(x_train))
np.random.shuffle(rand)
x_train=x_train[rand]
y_train=y_train[rand]
prob_train=prob_train[rand]


x_valid=x_train[0:60]
y_valid=y_train[0:60]
prob_valid=prob_train[0:60]


x_train=x_train[60::]
y_train=y_train[60::]
prob_train=prob_train[60::]



#layers
input_shape = Input(shape=(10,256,256,1))  
input_shape2 = Input(shape=(10,3)) 
 
conv1 = Conv2D(64, (3,3), activation = 'relu')
conv1=TimeDistributed(conv1)(input_shape)

batch1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
batch1=TimeDistributed(batch1)(conv1)


pool1 = MaxPooling2D((2, 2))
pool1=TimeDistributed(pool1)(batch1)



conv2 = Conv2D(128, (3,3), strides = 1, activation = 'relu')
conv2=TimeDistributed(conv2)(pool1)


conv3 = Conv2D(64, (3,3), strides = 1, activation = 'relu')
conv3=TimeDistributed(conv3)(conv2)

reshaped = Reshape((-1,64))
reshaped=TimeDistributed(reshaped)(conv3)

squashed_output = Lambda(squash)
squashed_output=TimeDistributed(squashed_output)(reshaped)


 
capsule = Capsule(8, 16, 3, True)

capsule=TimeDistributed(capsule)(squashed_output)



 
capsule2 = Capsule(3, 16, 3, True)
capsule2=TimeDistributed(capsule2)(capsule)


output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))
output=TimeDistributed(output)(capsule2)

output_prob = Multiply()([input_shape2,output])

final_outputs=GlobalMaxPooling1D()(output_prob)




#defina model
model = Model(inputs=[input_shape,input_shape2], outputs=[final_outputs])

adam = optimizers.Adam(lr=1e-4) 
model.compile(loss=margin_loss, optimizer=adam, metrics=['accuracy'])

#callback
filepath="path\weights-improvement-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#class weight
from sklearn.utils.class_weight import compute_class_weight

y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))
########################################

#Train
model.fit([x_train,prob_train],[y_train], batch_size = 8, epochs = 150, validation_data = ([x_valid,prob_valid],[y_valid]),callbacks=callbacks_list,shuffle=True,class_weight=d_class_weights)
