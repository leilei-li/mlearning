import tensorflow as tf
import numpy as np
import random


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_data():
    batch_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    sample_x = random.sample(batch_x, 2)
    sample_y = []
    for data in sample_x:
        if data == [0, 0]:
            sample_y.append([1, 0])
        if data == [0, 1]:
            sample_y.append([0, 1])
        if data == [1, 0]:
            sample_y.append([0, 1])
        if data == [1, 1]:
            sample_y.append([1, 0])

    return np.array(sample_x), np.array(sample_y)


x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
w1 = weight_variable(shape=[2, 2])
b1 = weight_variable(shape=[2])
xw1_plus_b1 = tf.nn.relu(tf.matmul(x, w1) + b1)
w2 = weight_variable(shape=[2, 2])
b2 = bias_variable(shape=[2])
prediction = tf.nn.softmax(tf.matmul(xw1_plus_b1, w2) + b2)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))  # 损失函数为交叉熵
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(50000):
        batch_x, batch_y = get_data()
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        if i % 50 == 0:
            print(sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y}))
            train_accuacy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
            print("step %d, training accuracy %g" % (i, train_accuacy))
    print("test accuracy %g" % (
        accuracy.eval(
            feed_dict={x: np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                       y_: np.array([[1, 0], [0, 1], [0, 1], [1, 0]])})))
