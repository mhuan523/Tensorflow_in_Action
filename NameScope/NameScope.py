# coding: utf-8
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime

housing = fetch_california_housing()
m, n = housing.data.shape
print("dataset: {} rows {} cols".format(m, n))

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = '/tmp'
log_dir = '{}/run-{}'.format(root_logdir, now)

scaler = StandardScaler()
housing_data_plus_bias_scaled = np.c_[np.ones((m, 1)), scaler.fit_transform(housing.data)]

X = tf.placeholder(tf.float32, [None, n + 1], name='X')
y = tf.placeholder(tf.float32, [None, 1], name='y')

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1, 1, tf.float32, seed=100), name='theta')
y_pred = tf.matmul(X, theta, name='prediction')

with tf.name_scope('loss') as scope:
    error = y - y_pred
    mse = tf.reduce_mean(tf.square(error), name='mse')

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = housing_data_plus_bias_scaled[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
            step = epoch * n_batches + batch_index
            file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()
    file_writer.flush()
    file_writer.close()
    print(best_theta)
