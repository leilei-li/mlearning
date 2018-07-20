import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 1])
y_lable = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.random_normal([1, 1]))
b = tf.Variable(tf.constant(0.1))
y = tf.matmul(x, w) + b
loss = tf.losses.mean_squared_error(y, y_lable)
train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(10000):
    x_in = np.random.rand(1, 1)
    y_in = 0.5 * x_in + 0.6
    w_, b_, loss_, train_step_ = sess.run([w, b, loss, train_step], feed_dict={x: x_in, y_lable: y_in})
    if i % 50 == 0:
        print(i, w_, b_, loss_, train_step_)
sess.close()
