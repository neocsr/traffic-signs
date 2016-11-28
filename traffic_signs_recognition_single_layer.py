"""
Traffic Sign Recognition using a Single Layer
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import cv2
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

start_time = time.time()

training_file = 'data/train.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as file:
    train = pickle.load(file)

with open(testing_file, mode='rb') as file:
    test = pickle.load(file)

print(train.keys())  # dict_keys(['labels', 'coords', 'features', 'sizes'])

X_train, X_valid, y_train, y_valid = train_test_split(train['features'],
                                                      train['labels'],
                                                      test_size=0.05,
                                                      random_state=123)
X_test, y_test = test['features'], test['labels']

sign_names = pd.read_csv('signnames.csv')
labels = sign_names.SignName.to_dict()
print(labels[0])      # 'Speed limit (20km/h)'

print(type(X_train))  # numpy.ndarray
print(X_train.shape)  # (39209, 32, 32, 3)

print(type(X_valid))    # numpy.ndarray
print(X_valid.shape)  # (39209, 32, 32, 3)

print(type(X_test))   # numpy.ndarray
print(X_test.shape)  # (39209, 32, 32, 3)

print(type(y_train))  # numpy.ndarray
print(y_train.shape)  # (39209,)

n_train = y_train.size
n_valid = y_valid.size
n_test = y_test.size

image_shape = X_train[0].shape
print(image_shape)  # (32, 32, 3)

n_classes = len(set(y_train))
print(n_classes)  # 43

print("Number of training examples =", n_train)    # 39209
print("Number of validation examples =", n_valid)  # 39209
print("Number of testing examples =", n_test)      # 12630
print("Image data shape =", image_shape)           # (32, 32, 3)
print("Number of classes =", n_classes)            # 43

# A single pixel
image = X_train[0]
print("A single pixel: ", image[0][0])  # [75 78 80]

# Data exploration visualization goes here.
# Feel free to use as many code cells as needed.
# label = labels[labels.ClassId == y_train[0]].SignName[0]
# plt.figure('Color')
# plt.imshow(image)
# plt.axis('off')
# plt.title(label)
# plt.show(block=False)

# Grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.figure('Grayscale')
# plt.imshow(gray_image, cmap='gray')
# plt.axis('off')
# plt.title(label)
# plt.show(block=False)


# plt.show()


# Design and Test a Model Architecture

# Preprocess the data
# Grayscale
def grayscale(images):
    n_images = images.shape[0]
    converted = []
    for i in range(0, n_images):
        converted.append(cv2.cvtColor(X_train[i], cv2.COLOR_BGR2GRAY))
    return np.array(converted)


# Normalize Grayscale Image using Min-Max
def normalize(images):
    x_min = images.min()
    x_max = images.max()
    a = 0.1
    b = 0.9
    return a + (images - x_min)*(b - a)/(x_max - x_min)


def norm(images):
    return normalize(grayscale(images))


X_train_norm = norm(X_train).reshape(X_train.shape[0], 1024)
X_valid_norm = norm(X_valid).reshape(X_valid.shape[0], 1024)
X_test_norm = norm(X_test).reshape(X_test.shape[0], 1024)

encoder = LabelBinarizer()
encoder.fit(y_train)

y_train_one_hot = encoder.transform(y_train)
y_valid_one_hot = encoder.transform(y_valid)
y_test_one_hot = encoder.transform(y_test)

# Parameters
epochs = 10
batch_size = 512
learning_rate = 0.01

n_input = 32*32                # 1024
n_classes = len(set(y_train))  # 43

# Input: (k, 32*32)
x = tf.placeholder(tf.float32, [None, n_input])

# Output: (k, 43)
y = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.truncated_normal([n_input, n_classes]))
biases = tf.Variable(tf.zeros([n_classes], dtype=tf.float32))

# Feed dicts for training, validation, and test session
train_feed_dict = {x: X_train_norm, y: y_train_one_hot}
valid_feed_dict = {x: X_valid_norm, y: y_valid_one_hot}
test_feed_dict = {x: X_test_norm, y: y_test_one_hot}

# Model
logits = tf.matmul(x, weights) + biases
logits = -np.amax(logits)
pred = tf.nn.softmax(logits)

cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)),
                               reduction_indices=1)
# cross_entropy = -tf.reduce_sum(y * tf.log(pred), reduction_indices=1)
loss = tf.reduce_mean(cross_entropy)

# Evaluate Mode
predicted_y = tf.argmax(pred, dimension=1)
expected_y = tf.argmax(y, dimension=1)

correct_pred = tf.equal(predicted_y, expected_y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# The accuracy measured against the validation set
valid_acc = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []
test_acc_batch = []

init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(X_train_norm)/batch_size))

    for epoch_i in range(epochs):

        # The training cycle
        for batch_i in range(batch_count):
            # Get a batch of training features and labels
            start = batch_i*batch_size
            batch_features = X_train_norm[start:start+batch_size]
            batch_labels = y_train_one_hot[start:start+batch_size]

            # Run optimizer and get loss
            _, l = session.run([optimizer, loss],
                               feed_dict={x: batch_features, y: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                train_acc = session.run(accuracy, feed_dict=train_feed_dict)
                valid_acc = session.run(accuracy, feed_dict=valid_feed_dict)
                test_acc = session.run(accuracy, feed_dict=test_feed_dict)
                print('training: {:.4f}  '.format(train_acc),
                      'validation: {:.4f}  '.format(valid_acc),
                      'testing: {:.4f}  '.format(test_acc),
                      'loss: {:.4f}  '.format(l))

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(train_acc)
                valid_acc_batch.append(valid_acc)
                test_acc_batch.append(test_acc)

    test_acc = session.run(accuracy, feed_dict=test_feed_dict)
    print('Testing accuracy {:.4f}'.format(test_acc))

end_time = time.time()
print('Duration: {:.0f}m'.format((end_time - start_time) / 60))

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
acc_plot.plot(batches, test_acc_batch, 'c', label='Testing Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()
plt.show()
