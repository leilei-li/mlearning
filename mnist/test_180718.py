import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnistData", one_hot=True)
in_data = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(in_data, [-1, 28, 28, 1])
y_lable = tf.placeholder(tf.float32, [None, 10])

# conv1 = tf.layers.conv2d(inputs=x_image, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
# pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)
# conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
# pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
# pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])
# fc1 = tf.layers.dense(inputs=pool2, units=1024, activation=tf.nn.relu)
# dropout = tf.layers.dropout(inputs=fc1)
# fc2 = tf.layers.dense(inputs=fc1, units=10, activation=tf.nn.softmax)
# y = fc2

y = tf.layers.dense(inputs=in_data, units=10, activation=tf.nn.softmax)
lose_cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_lable, logits=y)
tf.summary.scalar('loss', lose_cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(lose_cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_lable, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
merge = tf.summary.merge_all()
summary_write = tf.summary.FileWriter(logdir='tf_log/', graph=tf.get_default_graph())
for i in range(2000):
    if i % 20 == 0:
        batch = mnist.test.next_batch(50)
        train_accuacy, loss = sess.run([accuracy, lose_cross_entropy],
                                       feed_dict={in_data: batch[0], y_lable: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuacy))
        print(loss)
    batch = mnist.train.next_batch(50)
    _, summary = sess.run([train_step, merge], feed_dict={in_data: batch[0], y_lable: batch[1]})
    summary_write.add_summary(summary, global_step=i)
summary_write.close()
sess.close()
