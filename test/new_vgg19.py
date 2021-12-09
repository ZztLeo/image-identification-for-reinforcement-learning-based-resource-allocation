import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage import io, transform
# define different layer functions
# we usually don't do convolution and pooling on batch and channel


def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):

    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def gen_data(path):
    imgs = []
    labels = []
    with open(path) as file:
        contents = file.readlines()
        for content in contents:
            img_name, label = content.split(' ')
            # print(img_name, label[0])
            label = label[0]
            img = io.imread('E:/graduate student/实验室/中兴-业务编排/test'+img_name)
            img = transform.resize(img, (224, 224, 1))
            imgs.append(img)
            labels.append(label)
    print(len(imgs), len(labels))
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def convLayer(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding = "SAME"):
    """convlutional"""
    channel = int(x.get_shape()[-1])
    print(channel)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b = tf.get_variable("b", shape = [featureNum])
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
        out = tf.nn.bias_add(featureMap, b)
        return tf.nn.relu(out,name=scope.name)

class VGG19(object):
    """VGG model"""
    def __init__(self, x, keepPro, classNum):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        # self.MODELPATH = modelPath
        #build CNN
        # self.buildCNN()

    def buildCNN(self):
        """build model"""
        conv1_1 = convLayer(self.X, 3, 3, 1, 1, 64, "conv1_1" )
        conv1_2 = convLayer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
        pool1 = maxPoolLayer(conv1_2, 2, 2, 2, 2, "pool1")

        conv2_1 = convLayer(pool1, 3, 3, 1, 1, 128, "conv2_1")
        conv2_2 = convLayer(conv2_1, 3, 3, 1, 1, 128, "conv2_2")
        pool2 = maxPoolLayer(conv2_2, 2, 2, 2, 2, "pool2")

        conv3_1 = convLayer(pool2, 3, 3, 1, 1, 256, "conv3_1")
        conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 256, "conv3_2")
        conv3_3 = convLayer(conv3_2, 3, 3, 1, 1, 256, "conv3_3")
        conv3_4 = convLayer(conv3_3, 3, 3, 1, 1, 256, "conv3_4")
        pool3 = maxPoolLayer(conv3_4, 2, 2, 2, 2, "pool3")

        conv4_1 = convLayer(pool3, 3, 3, 1, 1, 512, "conv4_1")
        conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
        conv4_3 = convLayer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")
        conv4_4 = convLayer(conv4_3, 3, 3, 1, 1, 512, "conv4_4")
        pool4 = maxPoolLayer(conv4_4, 2, 2, 2, 2, "pool4")

        conv5_1 = convLayer(pool4, 3, 3, 1, 1, 512, "conv5_1")
        conv5_2 = convLayer(conv5_1, 3, 3, 1, 1, 512, "conv5_2")
        conv5_3 = convLayer(conv5_2, 3, 3, 1, 1, 512, "conv5_3")
        conv5_4 = convLayer(conv5_3, 3, 3, 1, 1, 512, "conv5_4")
        pool5 = maxPoolLayer(conv5_4, 2, 2, 2, 2, "pool5")

        fcIn = tf.reshape(pool5, [-1, 7*7*512])
        fc6 = fcLayer(fcIn, 7*7*512, 4096, True, "fc6")
        dropout1 = dropout(fc6, self.KEEPPRO)

        fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc7, self.KEEPPRO)

        fc8 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")
        return fc8



if __name__ == '__main__':
    data, labels = gen_data('E:/graduate student/实验室/中兴-业务编排/test/train.txt')
    num_example = data.shape[0]

    arr = np.arange(num_example)
    # np.random.shuffle(arr)
    data = data[arr]
    label = labels[arr]
    #
    ratio = 0.8
    s = np.int(1013 * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]

    x = tf.placeholder(tf.float32, shape=[None, 224,224,1])
    x_img = tf.reshape(x,[-1,28,28,1])
    y = tf.placeholder(tf.float32, shape=[None, 9])

    vgg = VGG19(x_img, 1.0, 9)
    output = tf.nn.softmax(vgg.buildCNN())
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,labels=y))
    prediction = tf.argmax(output, 1)
    label = tf.argmax(y, 1)
    correct_prediction = tf.equal(prediction, label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        # mnist = input_data.read_data_sets('MNIST_data/')
        for i in range(101):
            for j in range(20):
                sess.run(optimize, feed_dict={x:x_train[j*40:(j+1)*40],y:convert_to_one_hot(y_train[j*40:(j+1)*40], 9)})
            if i % 10 == 0:
                print(sess.run(cost, feed_dict={x:x_train[j*40:(j+1)*40],y:convert_to_one_hot(y_train[j*40:(j+1)*40], 9)}))
            if i % 20 == 0:
                for k in range(len(x_val)):
                    print('TestSet Accuracy:', sess.run(accuracy, feed_dict={x:x_val,y:convert_to_one_hot(y_val,9)}))
        print('end')



