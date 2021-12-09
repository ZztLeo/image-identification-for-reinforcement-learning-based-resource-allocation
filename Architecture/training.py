import tensorflow as tf
import numpy as np
import inception_v3
import math
import matplotlib.pyplot as plt

def run_model(session, predict, loss_val, Xd, yd, epochs = 1, batch_size = 64, print_every = 100, training = None,
              plot_losses = False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)  # argmax(input,axis)  对比预测值与实际y值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterater over the dataset once
        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):  # 括号中计算出迭代次数
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx, :], y: yd[idx], is_training: training_now}
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict = feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt,
                                                                                                             loss,
                                                                                                             np.sum(
                                                                                                                 corr) / actual_batch_size))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_correct






num_classes=20
batch_size=32
epochs=100

X=tf.placeholder(tf.float32, [None, 299, 299, 3])
y=tf.placeholder(tf.int64, [None])
is_training=tf.placeholder(tf.bool)

y_out=inception_v3.inception_v3(X,y,is_training = is_training)

mean_loss=None
optimizer=None

# define loss
total_loss=tf.losses.softmax_cross_entropy(tf.one_hot(y,num_classes),logits = y_out,scope = 'softmax_loss')
mean_loss=tf.reduce_mean(total_loss)

# define optimizer   decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)
learning_rate=tf.train.exponential_decay(learning_rate = 0.045,global_step =3,decay_steps =4,decay_rate = 0.94)
optimizer=tf.train.GradientDescentOptimizer(learning_rate)