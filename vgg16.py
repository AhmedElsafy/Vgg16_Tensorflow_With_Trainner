# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from scipy.spatial import distance
import os
from __future__ import division


os.chdir('/home/mario/Downloads/Very_Deep')
class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        #if weights is not None and sess is not None:
            #self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc7') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc8') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 5],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[5], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]



    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i ==len(keys)-2:
                 sess.run(self.parameters[i].assign(weights[k][:,0:5]))
            #print(np.shape(weights[k])
            elif i ==len(keys)-1:
                 sess.run(self.parameters[i].assign(weights[k][0:5]))
            elif k=='fc6_W':
                sess.run(self.parameters[i].assign(weights[k][0:512]))
            else:
                 sess.run(self.parameters[i].assign(weights[k]))

def intiatWieghts(weight_file):
     weights = np.load(weight_file)
     keys = sorted(weights.keys())
     
     for i, k in enumerate(keys):
         exec("%s = %d" % (k,2))
         R=(np.random.uniform(low=-1, high=1, size=(np.shape(weights[k]))))
         exec(k + " = "+"R")

     str_exec_save = "np.savez('intial_weights.npz',"    
     for i in keys:    
         str_exec_save += "%s = %s," % (i,i)
     str_exec_save += ")"
     exec(str_exec_save)    
         

def load_Data(path):
    label=[]
    data=[]
    TX=[]
    Ty1=[]
    tX=[]
    ty1=[]
    vX=[]
    vy1=[]
    print(path)
    for directory in os.listdir(path):
        print(directory)
        x=np.zeros(5,dtype='float32')
        x[int(directory)]=1.0
        for image in os.listdir(path+directory):
            print(image)
            im=imread(path+directory+'/'+image,mode='RGB')
            im=imresize(im,(32,32))
            im=im.astype(float32)
            data.append(im)
            label.append(x)
    arr = np.array(np.arange(len (data)))
    np.random.shuffle(arr)
    
    print('Organizing the data')
    for k in arr[0:int(len(data)*0.6)]:
        TX.append(data[k])
        Ty1.append(label[k])
        
    for k in arr[int(len (arr)*0.6):int(len (arr)*0.895)]:     
        tX.append(data[k])
        ty1.append(label[k])

    for k in arr[int(len (arr)*0.895):]:     
        vX.append(data[k])
        vy1.append(label[k])      

     
    return TX,tX,vX,Ty1,ty1,vy1


def train(py_x,sess,path):
    
    TX, tX, vX,Ty1,ty1,vy1 = load_Data(path)
    Features_Size=np.shape(TX)[1::]
    train_layers=['fc8','fc7']
    X = tf.placeholder("float32", None)
    Y = tf.placeholder("float32", [None,5])
    predict_op = (tf.nn.sigmoid(vgg.probs))
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=vgg.probs,labels=Y))
    gradients = tf.gradients(cost, var_list)
    gradients = list(zip(gradients, var_list))
    #cost = distance.euclidean(predict_op,Y)
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)
# construct an optimizer

    steps=1000
    #acc=[]
    totalCost=[]
    for i in range(steps):
        Averagec=[]
        print(i)
        for start, end in zip(range(0, len(TX), 64), range(64, len(TX)+1, 64)):
            sess.run(train_op, feed_dict={vgg.imgs:TX[start:end] , Y:Ty1[start:end]})
            #Averagec.append(c)  
        #totalCost.append(np.mean(Averagec))
        a = sess.run(predict_op, feed_dict={vgg.imgs: tX[0:1000], Y:ty1[0:1000]})
        test=(a == a.max(axis=1)[:,None]).astype(int)
        test=test.astype(float32)
        #acu=np.mean(for ind, i in enumerate(test): print(np.mean(ty1[ind])-i))
        acu=0        
        for ind,i in enumerate(test):
            if np.count_nonzero(ty1[ind]-i)>0:
                acu=acu+1
        acu=acu/1000        
        acc.append(acu)
        print(acu)
        
           
def SaveWeights(vgg,sess):
    listWeights=vgg.parameters
    values_toSave={}
    for i in listWeights:
          LayerName=i.name
          LayerWeights=i.eval(sess)
          values_toSave[LayerName]=LayerWeights
    np.savez('Save_Weights.npz',**values_toSave)

            

if __name__ == '__main__':
    
    path='/home/mario/Downloads/Very_Deep/train_data/'
    
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None,32, 32, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    #vgg = vgg16(imgs, 'intial_weights.npz', sess)
    
    img1 = imread('/home/mario/Downloads/Very_Deep/arm2.jpeg', mode='RGB')
    img1 = imresize(img1, (32, 32))
    
    train(vgg16,sess)    
    

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})
    preds = (np.argsort(prob)[::-1])[0:5]
#    for p in preds:
#        print (class_names[p])
