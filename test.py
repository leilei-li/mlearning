import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist/mnistData", one_hot=True)


# define weight
def weight_value(shape):
    inital = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(inital)


# define bias
def bias_value(shape):
    inital = tf.constant(shape=shape, value=0.1)
    return tf.Variable(inital)


# define convolution function
def conv2d(x, w):
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME')


# define max_pool function
def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define the input layer
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])  # batch=-1.height=28,width=28,channel=1
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# define the first convolution layer
w1 = weight_value(
    shape=[5, 5, 1, 32])  # the size of patch(the kenerl of the convolution) is 5x5, in_channel=1,out_channel=32
b1 = bias_value(shape=[32])
conv1 = tf.nn.relu(conv2d(x_image, w1) + b1)
pool1 = max_pool_2x2(conv1)

# define the second convolution layer
w2 = weight_value(shape=[5, 5, 32, 64])
b2 = bias_value(shape=[64])
conv2 = tf.nn.relu(conv2d(pool1, w2) + b2)
pool2 = max_pool_2x2(conv2)

# define the fully connection layer, to avoid overfitting, define a drop_out function
w3 = weight_value(shape=[7 * 7 * 64, 1024])
b3 = bias_value(shape=[1024])
pool2_reshape = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])
keep_prob = tf.placeholder(dtype=tf.float32)
full_connect_out = tf.nn.dropout(tf.nn.relu(tf.matmul(pool2_reshape, w3) + b3), keep_prob=keep_prob)

# define the output layer,
w4 = weight_value(shape=[1024, 10])
b4 = bias_value(shape=[10])
out_result = tf.nn.softmax(tf.matmul(full_connect_out, w4) + b4)

# define the loss function, which is a cross-entropy
loss = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(out_result, 1e-10, 1.0)))

# define the train_step, learning rate is alpha=0.1, optimizer is stochastic gradient descent algorithm
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# define the accuary
accuary = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out_result, 1), tf.argmax(y_, 1)), dtype=tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
        if i % 50 == 0:
            print(sess.run([loss, accuary], feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0}))
