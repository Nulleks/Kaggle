# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 01:22:42 2018

@author: 0x
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
from random import shuffle
from tqdm import tqdm



n_features = 18
n_nodes_hl1 = 6 
n_nodes_hl2 = 6
num_classes = 1


def initialize_parameters():
    W1 = tf.get_variable(name='W1',shape=(n_features,n_nodes_hl1),dtype=tf.float32)
    b1 = tf.get_variable(name='b1',shape=(n_nodes_hl1,),dtype=tf.float32)
    W2 = tf.get_variable(name='W2',shape=(n_nodes_hl1,n_nodes_hl2),dtype=tf.float32)
    b2 = tf.get_variable(name='b2',shape=(n_nodes_hl2,),dtype=tf.float32)
    
    #output
    W3 = tf.get_variable(name='W3',shape=(n_nodes_hl2,num_classes),dtype=tf.float32)
    b3 = tf.get_variable(name='b3',shape=(num_classes,),dtype=tf.float32)
    parameters = {"W1" : W1, "b1": b1 , "W2": W2, "b2": b2 ,  "W3": W3, "b3": b3}
    
    return parameters

def create_placeholders(n_X):
    X = tf.placeholder(tf.float32, shape = (None, n_X))
    Y = tf.placeholder(tf.float32)
    
    return X, Y

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



def forward_propagation(X):
    h1_layer = tf.layers.dense(X, n_nodes_hl1, activation=tf.nn.relu, name="h1")
    h2_layer = tf.layers.dense(h1_layer, n_nodes_hl2, activation=tf.nn.relu, name="h2")
    logits = tf.layers.dense(h2_layer, num_classes, name="output")    
    return logits


def forward_propagationV2(data, parameters):    
    
    l1 = tf.add(tf.matmul(data, parameters["W1"]), parameters["b1"])
    l1 = tf.nn.relu(l1)
    # l1 = tf.nn.dropout(l1, keep_prob)

    l2 = tf.add(tf.matmul(l1, parameters["W2"]), parameters["b2"])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2, parameters["W3"]), parameters["b3"])
    #output = tf.nn.softmax(output)

    return output


def forward_prop(data):
    W1 = tf.get_variable(name='W1',shape=(n_features,n_nodes_hl1),dtype=tf.float32)
    b1 = tf.get_variable(name='b1',shape=(n_nodes_hl1,),dtype=tf.float32)
    W2 = tf.get_variable(name='W2',shape=(n_nodes_hl1,n_nodes_hl2),dtype=tf.float32)
    b2 = tf.get_variable(name='b2',shape=(n_nodes_hl2,),dtype=tf.float32)
    
    #output
    W3 = tf.get_variable(name='W3',shape=(n_nodes_hl2,num_classes),dtype=tf.float32)
    b3 = tf.get_variable(name='b3',shape=(num_classes,),dtype=tf.float32)
    
    l1 = tf.add(tf.matmul(data, W1), b1)
    l1=tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, W2), b2)
    l2=tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, W3), b3)
    
    return l3
    

def compute_cost(ZL, Y, num_clas):
    if num_clas == 1:
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = ZL, labels = Y))
    else:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ZL, labels = Y))
    
    return cost

def predict(X, parameters):
    Yhat, _ = forward_propagationV2(X, parameters)
    Yhat = np.argmax(Yhat, axis = 0)
    
    return Yhat


def model(X_train, Y_train, x_test, y_test, learning_rate = 0.009, num_epochs = 50, minibatch_size = 24, print_cost = True):
    ops.reset_default_graph()
    (m, n_x) = X_train.shape # esto tiene que ser num columnas
    costs = []    
    X, Y = create_placeholders(n_x)
    parameters = initialize_parameters()
    ZL = forward_propagationV2(X, parameters)
    cost = compute_cost(ZL, Y, num_classes)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    #saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m/ minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})                
                minibatch_cost += temp_cost / num_minibatches
                
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
                
        #saver.save(sess, './my_test_model')
        
        parameters = sess.run(parameters)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()
        
        predict_op = tf.argmax(ZL, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: x_test, Y: y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        
        #results = sess.run( forward_propagationV2(X, parameters), feed_dict={X: x_test[1].astype("float32").reshape(1,-1)} )
        results = sess.run( forward_propagationV2(X, parameters), feed_dict={X: x_test.astype("float32")} )
        #results = np.argmax(results, axis = 0)
        #print(results)
        
        return train_accuracy, test_accuracy, parameters, results
 