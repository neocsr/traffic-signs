"""
Multilayer Perceptron

https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/multilayer_perceptron.ipynb
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist/', one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_input = 784     # mnist data input (28*28)
n_classes = 10    # mnist total classes (0-9 digits)

# TensorFlow Graph

# Input x:  (k, 784)
x = tf.placeholder(dtype='float', shape=[None, n_input], name='x')

# Output y: (k, 10)
y = tf.placeholder(dtype='float', shape=[None, n_classes], name='y')


# Model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    # layer_1: (k, 256)
    with tf.name_scope('layer_1'):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    # layer_2: (k, 256)
    with tf.name_scope('layer_2'):
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    # out_layer: (k, 10)
    with tf.name_scope('output'):
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer

# Store layers weight and bias

# Weights
# h1:  (784, 256)
# h2:  (256, 256)
# out: (256, 10)
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

# Biases
# b1:  (256, 1)
# b2:  (256, 1)
# out: (10, 1)
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct Model
pred = multilayer_perceptron(x, weights, biases)

# Define Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize Variables
init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

        avg_cost += c / total_batch

    if epoch % display_step == 0:
        print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))

print('Optimization Finished!')

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

with sess.as_default():
    acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print('Accuracy:', acc)

sess.close()
