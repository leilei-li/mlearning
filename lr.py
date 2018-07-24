import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32, [None, 1])
y_lable = tf.placeholder(tf.float32, [None, 1])
# 一次函数，可以求出k=w, b
# w = tf.Variable(tf.random_normal([1, 1]))
# b = tf.Variable(tf.constant(0.1))
# y = tf.matmul(x, w) + b
# y = tf.nn.relu(y)
# 二次函数，全连接接隐藏层，这样的缺点就是无法得到权重和偏置，因为不再是线性的函数了
fc1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
ffc = tf.layers.dense(inputs=fc1, units=1024, activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=ffc, units=1)
loss = tf.losses.mean_squared_error(y_lable, fc2)
train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
x_in = np.reshape(np.linspace(-7, 7, 300), [-1, 1])
noise = np.random.normal(0, 0.05, x_in.shape)
y_in = np.sin(x_in) + noise
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_in, y_in)
plt.ion()
plt.show()
for i in range(10000):
    # w_, b_, loss_, train_step_ = sess.run([w, b, loss, train_step], feed_dict={x: x_in, y_lable: y_in})
    loss_, train_step_ = sess.run([loss, train_step], feed_dict={x: x_in, y_lable: y_in})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        # print(i, w_, b_, loss_, train_step_)
        print(i, loss_, train_step_)
        fc2_ = sess.run(fc2, feed_dict={x: x_in, y_lable: y_in})
        lines = ax.plot(x_in, fc2_, 'r-', lw=5)
        plt.pause(1)

sess.close()
