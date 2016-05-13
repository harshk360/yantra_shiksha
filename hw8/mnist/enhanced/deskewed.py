import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# (10000,784) => (10000,28,28,1)
x_image = tf.reshape(x, [-1,28,28,1])

#conv1
W_conv1 = weight_variable([5, 5, 1, 4])
b_conv1 = bias_variable([4])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print ("h_conv1" + str(h_conv1.get_shape()))

#conv2
W_conv2 = weight_variable([5, 5, 4, 8])
b_conv2 = bias_variable([8])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
print ("h_conv2" + str(h_conv2.get_shape()))

#conv3
W_conv3 = weight_variable([5, 5, 8, 16])
b_conv3 = bias_variable([16])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
print ("h_pool3" + str(h_pool3.get_shape()))

#conv4
W_conv4 = weight_variable([5, 5, 16, 32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
print ("h_conv4" + str(h_conv4.get_shape()))

#conv5
W_conv5 = weight_variable([5, 5, 32, 64])
b_conv5 = bias_variable([64])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
print ("h_conv5" + str(h_conv5.get_shape()))

h_pool5 = max_pool_2x2(h_conv5)
print ("h_pool5" + str(h_pool5.get_shape()))

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool5_flat = tf.reshape(h_pool5, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
merged = tf.merge_all_summaries()
sess = tf.Session(config = config)
test_writer = tf.train.SummaryWriter('/home/ubuntu/mnist_logs' + '/test', sess.graph)
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:  # Record test-set accuracy
        print(i)
    #     summary, acc = sess.run([merged, accuracy], feed_dict={
    # x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    #     test_writer.add_summary(summary, i)
    #     print('Accuracy at step %s: %s' % (i, acc))
    else: # train
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))