import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

import pdb

def base_network(img, training, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        conv1 = tf.layers.conv2d(inputs=img,
                                  filters=8,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation= None,
                                  name='conv1')
        batch_norm1 = tf.layers.batch_normalization(conv1, training = training)
        conv1 = tf.nn.relu(batch_norm1)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                        pool_size=[2, 2], 
                                        strides=2,
                                        name='pool1')
        
        conv2 = tf.layers.conv2d(inputs=pool1,
                                  filters=16,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation= None,
                                  name='conv2')
        batch_norm2 = tf.layers.batch_normalization(conv2, training = training)
        conv2 = tf.nn.relu(batch_norm2)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, 
                                        pool_size=[2, 2], 
                                        strides=2,
                                        name='pool2')
        
           
        conv3 = tf.layers.conv2d(inputs=pool2,
                                  filters=32,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation= None,
                                  name='conv3')
        batch_norm3 = tf.layers.batch_normalization(conv3, training = training)
        conv3 = tf.nn.relu(batch_norm3)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, 
                                        pool_size=[2, 2], 
                                        strides=2,
                                        name='pool3')

           
        conv4 = tf.layers.conv2d(inputs=pool3,
                                  filters=64,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation= None,
                                  name='conv4')
        batch_norm4 = tf.layers.batch_normalization(conv4, training = training)
        conv4 = tf.nn.relu(batch_norm4)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, 
                                        pool_size=[2, 2], 
                                        strides=2,
                                        name='pool4')

        conv5 = tf.layers.conv2d(inputs=pool4,
                                  filters=128,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation= None,
                                  name='conv5')
        batch_norm5 = tf.layers.batch_normalization(conv5, training = training)
        conv5 = tf.nn.relu(batch_norm5)
        
        return conv5

def intermediate_layer(img, training, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                conv = tf.layers.conv2d(inputs=img,
                                  filters=128,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation= None,
                                  name='conv',
                                  kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1),
                                  bias_initializer=tf.constant_initializer(0.1))
                batch_norm = tf.layers.batch_normalization(conv, training = training)
                conv = tf.nn.relu(batch_norm)
             
                return conv

def clf_layer(img, training, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                conv = tf.layers.conv2d(inputs=img,
                                        filters=1,
                                        kernel_size=[1, 1],
                                        padding='SAME', 
                                        activation=None, 
                                        name='conv')
                return conv

def reg_layer(img, training, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                conv = tf.layers.conv2d(inputs=img,
                                        filters=4,
                                        kernel_size=[1, 1],
                                        padding='SAME', 
                                        activation=None, 
                                        bias_initializer=tf.constant_initializer([4,4,8,8]),
                                        name='conv')
                return conv

def faster_rcnn(img, training, scope, n_classes):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                conv1 = tf.layers.conv2d(inputs=img,
                                  filters=128,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation= None,
                                  name='conv1')
                batch_norm1 = tf.layers.batch_normalization(conv1, training = training)
                conv1 = tf.nn.relu(batch_norm1)

                conv2 = tf.layers.conv2d(inputs=conv1,
                                  filters=128,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation= None,
                                  name='conv2')
                batch_norm2 = tf.layers.batch_normalization(conv2, training = training)
                conv2 = tf.nn.relu(batch_norm2)
                stretched_dim = conv2.shape[1] * conv2.shape[2] * conv2.shape[3]
                conv2_stretched = tf.reshape(conv2, [-1, stretched_dim])
                logits = tf.layers.dense(conv2_stretched, n_classes, name='logits')
                            
                return logits

def mask_rcnn(img, training, scope, n_classes):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                conv1 = tf.layers.conv2d(inputs=img,
                                  filters=128,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation= None,
                                  name='conv1')
                batch_norm1 = tf.layers.batch_normalization(conv1, training = training)
                conv1 = tf.nn.relu(batch_norm1)

                conv2 = tf.layers.conv2d(inputs=conv1,
                                  filters=128,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation= None,
                                  name='conv2')
                batch_norm2 = tf.layers.batch_normalization(conv2, training = training)
                conv2 = tf.nn.relu(batch_norm2)
                
                conv3 = tf.layers.conv2d(inputs=conv2,
                                  filters=1,
                                  kernel_size=[1, 1],
                                  padding='SAME',
                                  activation= None,
                                  name='conv3')


                tc = tf.layers.conv2d_transpose(conv3,filters=1, kernel_size=[3,3],strides=(2, 2),padding='valid',bias_initializer=tf.zeros_initializer())

                tc2 = tf.layers.conv2d_transpose(tc,filters=1, kernel_size=[3,3],strides=(2, 2),padding='valid',bias_initializer=tf.zeros_initializer())

                tc3 = tf.layers.conv2d_transpose(tc2,filters=1, kernel_size=[3,3],strides=(1, 1),padding='valid',bias_initializer=tf.zeros_initializer())

                tc4 = tf.layers.conv2d_transpose(tc3,filters=1, kernel_size=[2,2],strides=(1, 1),padding='valid',bias_initializer=tf.zeros_initializer())

                     
                return tc4