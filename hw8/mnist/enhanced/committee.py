import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_helpers import deskew, width_normalization
import numpy as np

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


####################################################################
####################################################################

one_x = tf.placeholder(tf.float32, [None, 784])
one_y_ = tf.placeholder(tf.float32, [None, 10])

# (40000,784) => (40000,28,28,1)
one_x_image = tf.reshape(one_x, [-1,28,28,1])

one_W_conv1 = weight_variable([5, 5, 1, 32])
one_b_conv1 = bias_variable([32])
one_h_conv1 = tf.nn.relu(conv2d(one_x_image, one_W_conv1) + one_b_conv1)
one_h_pool1 = max_pool_2x2(one_h_conv1)

one_W_conv2 = weight_variable([5, 5, 32, 64])
one_b_conv2 = bias_variable([64])
one_h_conv2 = tf.nn.relu(conv2d(one_h_pool1, one_W_conv2) + one_b_conv2)
one_h_pool2 = max_pool_2x2(one_h_conv2)

one_W_fc1 = weight_variable([7 * 7 * 64, 1024])
one_b_fc1 = bias_variable([1024])

one_h_pool2_flat = tf.reshape(one_h_pool2, [-1, 7*7*64])
one_h_fc1 = tf.nn.relu(tf.matmul(one_h_pool2_flat, one_W_fc1) + one_b_fc1)

one_keep_prob = tf.placeholder(tf.float32)
one_h_fc1_drop = tf.nn.dropout(one_h_fc1, one_keep_prob)

one_W_fc2 = weight_variable([1024, 10])
one_b_fc2 = bias_variable([10])

one_y_conv=tf.nn.softmax(tf.matmul(one_h_fc1_drop, one_W_fc2) + one_b_fc2)

one_cross_entropy = tf.reduce_mean(-tf.reduce_sum(one_y_ * tf.log(one_y_conv), reduction_indices=[1]))
one_train_step = tf.train.AdamOptimizer(1e-4).minimize(one_cross_entropy)

####################################################################
####################################################################

two_x = tf.placeholder(tf.float32, [None, 784])
two_y_ = tf.placeholder(tf.float32, [None, 10])

# (40000,784) => (40000,28,28,1)
two_x_image = tf.reshape(two_x, [-1,28,28,1])

two_W_conv1 = weight_variable([5, 5, 1, 32])
two_b_conv1 = bias_variable([32])
two_h_conv1 = tf.nn.relu(conv2d(two_x_image, two_W_conv1) + two_b_conv1)
two_h_pool1 = max_pool_2x2(two_h_conv1)

two_W_conv2 = weight_variable([5, 5, 32, 64])
two_b_conv2 = bias_variable([64])
two_h_conv2 = tf.nn.relu(conv2d(two_h_pool1, two_W_conv2) + two_b_conv2)
two_h_pool2 = max_pool_2x2(two_h_conv2)

two_W_fc1 = weight_variable([7 * 7 * 64, 1024])
two_b_fc1 = bias_variable([1024])

two_h_pool2_flat = tf.reshape(two_h_pool2, [-1, 7*7*64])
two_h_fc1 = tf.nn.relu(tf.matmul(two_h_pool2_flat, two_W_fc1) + two_b_fc1)

two_keep_prob = tf.placeholder(tf.float32)
two_h_fc1_drop = tf.nn.dropout(two_h_fc1, two_keep_prob)

two_W_fc2 = weight_variable([1024, 10])
two_b_fc2 = bias_variable([10])

two_y_conv=tf.nn.softmax(tf.matmul(two_h_fc1_drop, two_W_fc2) + two_b_fc2)

two_cross_entropy = tf.reduce_mean(-tf.reduce_sum(two_y_ * tf.log(two_y_conv), reduction_indices=[1]))
two_train_step = tf.train.AdamOptimizer(1e-4).minimize(two_cross_entropy)

####################################################################
####################################################################

three_x = tf.placeholder(tf.float32, [None, 784])
three_y_ = tf.placeholder(tf.float32, [None, 10])

# (40000,784) => (40000,28,28,1)
three_x_image = tf.reshape(three_x, [-1,28,28,1])

three_W_conv1 = weight_variable([5, 5, 1, 32])
three_b_conv1 = bias_variable([32])
three_h_conv1 = tf.nn.relu(conv2d(three_x_image, three_W_conv1) + three_b_conv1)
three_h_pool1 = max_pool_2x2(three_h_conv1)

three_W_conv2 = weight_variable([5, 5, 32, 64])
three_b_conv2 = bias_variable([64])
three_h_conv2 = tf.nn.relu(conv2d(three_h_pool1, three_W_conv2) + three_b_conv2)
three_h_pool2 = max_pool_2x2(three_h_conv2)

three_W_fc1 = weight_variable([7 * 7 * 64, 1024])
three_b_fc1 = bias_variable([1024])

three_h_pool2_flat = tf.reshape(three_h_pool2, [-1, 7*7*64])
three_h_fc1 = tf.nn.relu(tf.matmul(three_h_pool2_flat, three_W_fc1) + three_b_fc1)

three_keep_prob = tf.placeholder(tf.float32)
three_h_fc1_drop = tf.nn.dropout(three_h_fc1, three_keep_prob)

three_W_fc2 = weight_variable([1024, 10])
three_b_fc2 = bias_variable([10])

three_y_conv=tf.nn.softmax(tf.matmul(three_h_fc1_drop, three_W_fc2) + three_b_fc2)

three_cross_entropy = tf.reduce_mean(-tf.reduce_sum(three_y_ * tf.log(three_y_conv), reduction_indices=[1]))
three_train_step = tf.train.AdamOptimizer(1e-4).minimize(three_cross_entropy)

####################################################################
####################################################################

four_x = tf.placeholder(tf.float32, [None, 784])
four_y_ = tf.placeholder(tf.float32, [None, 10])

# (40000,784) => (40000,28,28,1)
four_x_image = tf.reshape(four_x, [-1,28,28,1])

four_W_conv1 = weight_variable([5, 5, 1, 32])
four_b_conv1 = bias_variable([32])
four_h_conv1 = tf.nn.relu(conv2d(four_x_image, four_W_conv1) + four_b_conv1)
four_h_pool1 = max_pool_2x2(four_h_conv1)

four_W_conv2 = weight_variable([5, 5, 32, 64])
four_b_conv2 = bias_variable([64])
four_h_conv2 = tf.nn.relu(conv2d(four_h_pool1, four_W_conv2) + four_b_conv2)
four_h_pool2 = max_pool_2x2(four_h_conv2)

four_W_fc1 = weight_variable([7 * 7 * 64, 1024])
four_b_fc1 = bias_variable([1024])

four_h_pool2_flat = tf.reshape(four_h_pool2, [-1, 7*7*64])
four_h_fc1 = tf.nn.relu(tf.matmul(four_h_pool2_flat, four_W_fc1) + four_b_fc1)

four_keep_prob = tf.placeholder(tf.float32)
four_h_fc1_drop = tf.nn.dropout(four_h_fc1, four_keep_prob)

four_W_fc2 = weight_variable([1024, 10])
four_b_fc2 = bias_variable([10])

four_y_conv=tf.nn.softmax(tf.matmul(four_h_fc1_drop, four_W_fc2) + four_b_fc2)

four_cross_entropy = tf.reduce_mean(-tf.reduce_sum(four_y_ * tf.log(four_y_conv), reduction_indices=[1]))
four_train_step = tf.train.AdamOptimizer(1e-4).minimize(four_cross_entropy)

####################################################################
####################################################################

y_ = tf.placeholder(tf.float32, [None, 10])

y_add1 = tf.add(one_y_conv, two_y_conv)
y_add2 = tf.add(y_add1, three_y_conv)
y_add3 = tf.add(y_add2, four_y_conv)
y_conv = tf.div(y_add3, 4)

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

def width(image, width, index):
    if(batch_original[1][index][1]) == 1:
        return image
    else:
        return width_normalization(np.reshape(image, (28, 28)), width)

sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch_original = mnist.train.next_batch(50)
    width10_temp = [width(image, 10, idx) for idx, image in enumerate(batch_original[0])]
    width20_temp = [width(image, 20, idx) for idx, image in enumerate(batch_original[0])]
    deskew_temp = [deskew(np.reshape(image, (28, 28))) for image in batch_original[0]]
    if i % 100 == 0:  # Record test-set accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict={one_x: mnist.test.images,
                                                                two_x:mnist.test.images,
                                                                three_x:mnist.test.images,
                                                                four_x:mnist.test.images,
                                                                y_: mnist.test.labels,
                                                                one_keep_prob: 1.0,
                                                                two_keep_prob: 1.0,
                                                                three_keep_prob: 1.0,
                                                                four_keep_prob: 1.0})
        test_writer.add_summary(summary, i)
        print('Averaged Test Accuracy at step %s: %s' % (i, acc))
    else:# train
        #print (batch[0])
        sess.run(one_train_step, feed_dict={one_x: batch_original[0],
                                            one_y_: batch_original[1],
                                            one_keep_prob: 0.5})

        sess.run(two_train_step, feed_dict={two_x: width10_temp,
                                            two_y_: batch_original[1],
                                            two_keep_prob: 0.5})

        sess.run(three_train_step, feed_dict={three_x: width20_temp,
                                            three_y_: batch_original[1],
                                            three_keep_prob: 0.5})

        sess.run(four_train_step, feed_dict={four_x: deskew_temp,
                                            four_y_: batch_original[1],
                                            four_keep_prob: 0.5})


print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={one_x: mnist.test.images,
                                                                two_x:mnist.test.images,
                                                                three_x:mnist.test.images,
                                                                four_x:mnist.test.images,
                                                                y_: mnist.test.labels,
                                                                one_keep_prob: 1.0,
                                                                two_keep_prob: 1.0,
                                                                three_keep_prob: 1.0,
                                                                four_keep_prob: 1.0}))


