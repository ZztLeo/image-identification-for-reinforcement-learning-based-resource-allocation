import numpy as np
import tensorflow as tf
from skimage import io, transform


path = '/home/network/mahaoli/untitled1/data/train.txt'
w = 224
h = 224
c = 1

# def read_img(path):
#     cate = [path + x for x in os.listdir(path) if os.path.isdir(path)]
#     imgs = []
#     labels = []
#     for idx, folder in enumerate(cate):
#         for im in glob.glob(folder + '/*.jpg'):
#             print('reading the image: %s' % (im))
#             img = io.imread(im)
#             img = transform.resize(img, (w, h, c))
#             imgs.append(img)
#             labels.append(idx)
#     return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

def gen_data(path):
    imgs = []
    labels = []
    with open(path) as file:
        contents = file.readlines()
        for content in contents:
            img_name, label = content.split(' ')
            # print(img_name, label[0])
            label = label[0]
            img = io.imread('/home/network/mahaoli/untitled1/data/'+img_name)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(label)
    print(len(imgs), len(labels))
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
data, labels = gen_data(path)

num_example = data.shape[0]
arr = np.arange(num_example)
# np.random.shuffle(arr)
data = data[arr]
label = labels[arr]

ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val   = data[s:]
y_val   = label[s:]

def build_network(height, width, channel):
    x = tf.placeholder(tf.float32, shape=[None, height, width, channel], name='input')
    y = tf.placeholder(tf.int64, shape=[None, 9], name='labels_placeholder')

    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(input, w):
        return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')

    def pool_max(input):
        return tf.nn.max_pool(input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

    def fc(input, w, b):
        return tf.matmul(input, w) + b

    # conv1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, 1, 64])
        biases = bias_variable([64])
        output_conv1_1 = tf.nn.relu(conv2d(x, kernel) + biases, name=scope)

    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 64, 64])
        biases = bias_variable([64])
        output_conv1_2 = tf.nn.relu(conv2d(output_conv1_1, kernel) + biases, name=scope)

    pool1 = pool_max(output_conv1_2)

    # conv2
    with tf.name_scope('conv2_1') as scope:
        kernel = weight_variable([3, 3, 64, 128])
        biases = bias_variable([128])
        output_conv2_1 = tf.nn.relu(conv2d(pool1, kernel) + biases, name=scope)

    with tf.name_scope('conv2_2') as scope:
        kernel = weight_variable([3, 3, 128, 128])
        biases = bias_variable([128])
        output_conv2_2 = tf.nn.relu(conv2d(output_conv2_1, kernel) + biases, name=scope)

    pool2 = pool_max(output_conv2_2)

    # conv3
    with tf.name_scope('conv3_1') as scope:
        kernel = weight_variable([3, 3, 128, 256])
        biases = bias_variable([256])
        output_conv3_1 = tf.nn.relu(conv2d(pool2, kernel) + biases, name=scope)

    with tf.name_scope('conv3_2') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_2 = tf.nn.relu(conv2d(output_conv3_1, kernel) + biases, name=scope)

    with tf.name_scope('conv3_3') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_3 = tf.nn.relu(conv2d(output_conv3_2, kernel) + biases, name=scope)

    pool3 = pool_max(output_conv3_3)

    # conv4
    with tf.name_scope('conv4_1') as scope:
        kernel = weight_variable([3, 3, 256, 512])
        biases = bias_variable([512])
        output_conv4_1 = tf.nn.relu(conv2d(pool3, kernel) + biases, name=scope)

    with tf.name_scope('conv4_2') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv4_2 = tf.nn.relu(conv2d(output_conv4_1, kernel) + biases, name=scope)

    with tf.name_scope('conv4_3') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv4_3 = tf.nn.relu(conv2d(output_conv4_2, kernel) + biases, name=scope)

    pool4 = pool_max(output_conv4_3)

    # conv5
    with tf.name_scope('conv5_1') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_1 = tf.nn.relu(conv2d(pool4, kernel) + biases, name=scope)

    with tf.name_scope('conv5_2') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_2 = tf.nn.relu(conv2d(output_conv5_1, kernel) + biases, name=scope)

    with tf.name_scope('conv5_3') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_3 = tf.nn.relu(conv2d(output_conv5_2, kernel) + biases, name=scope)

    pool5 = pool_max(output_conv5_3)

    #fc6
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        kernel = weight_variable([shape, 4096])
        biases = bias_variable([4096])
        pool5_flat = tf.reshape(pool5, [-1, shape])
        output_fc6 = tf.nn.relu(fc(pool5_flat, kernel, biases), name=scope)

    #fc7
    with tf.name_scope('fc7') as scope:
        kernel = weight_variable([4096, 4096])
        biases = bias_variable([4096])
        output_fc7 = tf.nn.relu(fc(output_fc6, kernel, biases), name=scope)

    #fc8
    with tf.name_scope('fc8') as scope:
        kernel = weight_variable([4096, 9])
        biases = bias_variable([9])
        output_fc8 = tf.nn.relu(fc(output_fc7, kernel, biases), name=scope)

    finaloutput = tf.nn.softmax(output_fc8, name="softmax")

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=finaloutput, labels=y))
    optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    prediction_labels = tf.argmax(finaloutput, 1)
    read_labels = tf.argmax(y, 1)

    correct_prediction = tf.equal(prediction_labels, read_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    return dict(
        x=x,
        y=y,
        optimize=optimize,
        correct_prediction=correct_prediction,
        correct_times_in_batch=correct_times_in_batch,
        cost=cost,
        accuracy=accuracy
    )


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


def dropout(x, keepPro):
    """dropout"""
    return tf.nn.dropout(x, keepPro)


def train_network(graph, batch_size, num_epochs, pb_file_path):
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        for epoch_index in range(num_epochs):
            for i in range(batch_size):
                sess.run([graph['optimize']], feed_dict={
                    graph['x']: x_train[i*40:(i+1)*40],
                    graph['y']: (convert_to_one_hot(y_train[i*40:(i+1)*40],9))})
                print(sess.run([graph['cost']],feed_dict={graph['x']:x_train[i*40:(i+1)*40], graph['y']: (convert_to_one_hot(y_train[i*40:(i+1)*40],9))}))
            if epoch_index % 5 == 0:
                for k in range(len(x_val)):
                    print('TestSet Accuracy:',
                          sess.run(graph['accuracy'], feed_dict={graph['x']: x_val, graph['y']: convert_to_one_hot(y_val, 9)}))
        print('end')
                # print(sess.run(graph['y']))



def main():
    batch_size = 20
    num_epochs = 100

    pb_file_path = "vggs.pb"

    g = build_network(height=224, width=224, channel=1)
    train_network(g, batch_size, num_epochs, pb_file_path)

if __name__ == '__main__':
    main()
    # print(data.shape[1])


