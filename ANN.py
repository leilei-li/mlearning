import tensorflow as tf
import numpy as np
import random


# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("mnist/mnistData", one_hot=True)


def get_dataset(filename, number):
    f = open(filename)
    line = f.readline()
    line = f.readline()
    dataSet = []
    while (line):
        data = line.strip().split(' ')
        dd = []
        for d in data:
            dd.append(float(d))
        line = f.readline()
        y = int(float(line.strip()))
        dd.append(y)
        dataSet.append(dd)
        line = f.readline()
    if number > 0:
        dataSet = random.sample(dataSet, number)
    else:
        dataSet = dataSet
    batch_x = []
    batch_y = []
    for data in dataSet:
        x = data[:8]
        x = normalization_x(x)
        y = data[len(data) - 1]
        batch_x.append(np.array(x))
        if y == 1:
            batch_y.append(np.array([1, 0, 0]))
        elif y == 0:
            batch_y.append(np.array([0, 1, 0]))
        elif y == -1:
            batch_y.append(np.array([0, 0, 1]))
    return np.array(batch_x), np.array(batch_y)


def normalization_x(x_input):
    min = 9999999
    max = -9999999
    for x in x_input:
        if x > max:
            max = x
        if x < min:
            min = x
    normal_x = []
    for x in x_input:
        normal_x.append((x - min) / (max - min))
    return normal_x


def addLayer(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.zeros([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]))
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


x = tf.placeholder(dtype=tf.float32, shape=[None, 8])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])
# 隐藏层
w1 = weight_variable(shape=[8,8])
b1 = bias_variable(shape=[8])
xw1_plus_b1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# 输出层10个
w2 = weight_variable(shape=[8,3])
b2 = bias_variable(shape=[3])
prediction = tf.nn.softmax(tf.matmul(xw1_plus_b1, w2) + b2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(prediction + 1e-10))  # 损失函数为交叉熵
# MSVE = tf.reduce_mean(tf.square(y_ - prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        batch_x, batch_y = get_dataset('darkSoul/backstab_training_data.train', 50)
        # batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        if i % 50 == 0:
            train_accuacy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
            print(sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y}))
            print("step %d, training accuracy %g" % (i, train_accuacy))
    batch_x, batch_y = get_dataset('darkSoul/backstab_training_data.test', -1)
    print("test accuracy %g" % (accuracy.eval(feed_dict={x: batch_x, y_: batch_y})))
    # print("test accuracy %g" % (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
