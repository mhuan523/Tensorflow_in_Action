# conding: utf-8
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import numpy as np

housing = fetch_california_housing()
m, n = housing.data.shape

scaler = StandardScaler()
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
housing_data_plus_bias_scaled = scaler.fit_transform(housing_data_plus_bias)

n_epochs = 10000
learning_rate = 0.01
X = tf.constant(housing_data_plus_bias_scaled, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = tf.gradients(mse, theta)[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
