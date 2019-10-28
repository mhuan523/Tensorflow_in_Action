# coding: utf-8
import tensorflow as tf
from functools import partial
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    mnist = input_data.read_data_sets('/tmp/data')

    batch_norm_momentum = 0.9
    learning_rate = 0.1
    X = tf.placeholder(tf.float32, [None, n_inputs], name='X')
    y = tf.placeholder(tf.int32, None, name='y')
    training = tf.placeholder_with_default(False, None, name='training')

    with tf.name_scope('dnn'):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=batch_norm_momentum)
        my_dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)

        hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
        bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
        hidden2 = my_dense_layer(bn1, n_hidden2, name='hidden2')
        bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
        logists_before_bn = my_dense_layer(bn2, n_outputs, name='outputs')
        logists = my_batch_norm_layer(logists_before_bn)

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=
        logists)
        loss = tf.reduce_mean(xentropy, name='loss')

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logists, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    n_epoches = 20
    batch_size = 200

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epoches):
            for iteraton in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run([training_op, extra_update_ops],feed_dict={training: True, X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
            print(epoch, 'Test accuracy:', accuracy_val)