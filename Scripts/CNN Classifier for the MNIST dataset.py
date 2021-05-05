# code to build, compile and fit a convolutional neural network (CNN) model to the MNIST dataset of images of
# handwritten digits.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Run this cell first to import all required packages. Do not make any imports elsewhere in the notebook
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.utils import to_categorical
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

'''The MNIST dataset consists of a training set of 60,000 handwritten digits with corresponding labels, and a test set 
of 10,000 images. The images have been normalised and centred. The dataset is frequently used in machine learning 
research, and has become a standard benchmark for image classification models.

Your goal is to construct a neural network that classifies images of handwritten digits into one of 10 classes.'''
# Run this cell to load the MNIST data

mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()
print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))

# plot the digits
y = train_labels
X = train_images
fig, ax = plt.subplots(2, 5)
for i, ax in enumerate(ax.flatten()):
    im_idx = np.argwhere(y == i)[0]
    plottable_image = np.reshape(X[im_idx], (28, 28))
    ax.imshow(plottable_image, cmap='gray_r')

# pick a sample to plot
sample = 1
image = train_images[sample]
# plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()

# Complete the following function.
print(train_images.shape)  # (60000,28, 28)--number of samples, height of image, width of image

# frequency chart
unique, counts = np.unique(y, return_counts=True)
plt.bar(unique, counts)
plt.xticks(unique)
plt.xlabel("Digits")
plt.ylabel("Quantity")
plt.title("Digits in MNIST 784 dataset")
plt.show()


def scale_mnist_data(xtrain, xtest, ytrain, ytest):
    """
    Scales training and test set so that they have minimum and maximum values equal to 0 and 1 respectively (normalize).
    and reshapes so that there is a single channel """
    scaled_train_images = xtrain.reshape((xtrain.shape[0], 28, 28, 1))
    scaled_test_images = xtest.reshape((xtest.shape[0], 28, 28, 1))
    scaled_train_images, scaled_test_images = xtrain[..., np.newaxis] / 255.0, xtest[..., np.newaxis] / 255.0
    # one hot encode target values
    train_labels = to_categorical(ytrain)
    test_labels = to_categorical(ytest)
    return scaled_train_images, scaled_test_images, train_labels, test_labels


scaled_train_images, scaled_test_images, train_labels, test_labels = scale_mnist_data(train_images, test_images,
                                                                                      train_labels, test_labels)

# Input Layer expects data in the format of NHWC
# N = Number of samples
# H = Height of the Image
# W = Width of the Image
# C = Number of Channels

print(scaled_train_images.shape)  # (60000, 28, 28, 1)

'''
Build the convolutional neural network model
Using the Sequential API, build your CNN model according to the following spec:

The model should use the input_shape in the function argument to set the input size in the first layer.
A 2D convolutional layer with a 3x3 kernel and 8 filters. 
Use 'SAME' zero padding and ReLU activation functions. 
Make sure to provide the input_shape keyword argument in this first layer.

A max pooling layer, with a 2x2 window, and default strides.

A flatten layer, which unrolls the input into a one-dimensional tensor.

Two dense hidden layers, each with 64 units and ReLU activation functions.

A dense output layer with 10 units and the softmax activation function.

In particular, your neural network should have six layers.'''
# need this code to avoid "failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED" error
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

'''
#This model from the assignment is not as accurate as the model from machinelearningmaster.com
def get_model(input_shape):
    model = Sequential(
        [Conv2D(8, (3, 3), activation='relu', input_shape=input_shape, padding='SAME', name='Conv2D_layer_1'),
         # default data_format='channels_last'
         MaxPooling2D((2, 2), name='MaxPool_layer_2'),
         Flatten(name='Flatten_layer_3'),
         Dense(64, activation='relu', name='Dense_layer_4'),
         Dense(64, activation='relu', name='Dense_layer_5'),
         Dense(10, activation='softmax', name='Dense_layer_6')
         ])
    return model
    '''


# the assignment asks for the model above, but I want to compare my answers with machinelearningmastery.com
def get_model():
    model = Sequential(
        [Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1),
                name='Conv2D_layer_1'),
         # default data_format='channels_last'
         MaxPooling2D((2, 2), name='MaxPool_layer_2'),
         Flatten(name='Flatten_layer_3'),
         Dense(100, activation='relu', kernel_initializer='he_uniform', name='Dense_layer_4'),
         Dense(10, activation='softmax', name='Dense_layer_5')
         ])
    return model


model = get_model()
print(model.summary())  # this is easier to read

# compile the model

#     Optimizers in machine learning are used to tune the parameters of a neural network
#     in order to minimize the cost function.Contrary to what many believe, the loss function is not the same thing as
#     the cost function. While the loss function computes the distance of a single prediction from its actual value,
#     the cost function is usually more general. Indeed, the cost function can be, for example, the sum of loss
#     functions over the training set plus some regularization.

#     Compile the model using the Adam optimiser (with default settings),
#     the cross-entropy loss function and accuracy as the only metric."""
# SGD stochastic gradient descent Instead of computing the gradients over the entire dataset, it performs a
# parameter update for each example in the dataset. The problem of SGD is that the updates are frequent and with a
# high variance, so the objective function heavily fluctuates during training. Although this alows
# the function to jump to better local minima, it disadvantages the convergence in a specific local minima.
# a better solution would be Mini Batch Gradient Descent these gradient descent algorithms are not good for sparse data,
# subject to the choice of learning rate, and have a high possibility of getting stuck into a suboptimal local minima

# opt = SGD(lr=0.01, momentum=0.9)

# Adaptive optimizers on the other hand don't require a tuning of the learning rate
# value. They solve issues of gradient descents algorithms, by performing small updates for
# freq occuring features and large updates for the rarest ones
# ADAM is the best among adaptive optimizers in most cases, it is good with sparse data, and no
# need to focus on the learning rate.
opt = tf.keras.optimizers.Adam()
acc = tf.keras.metrics.Accuracy()


def compile_model(model):
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


compile_model(model)

# FIT THE MODEL TO THE TRAINING DATA############################
print(scaled_train_images.shape)


def train_model(scaled_train_images, train_labels, scaled_test_images, test_labels):
    history = model.fit(scaled_train_images, train_labels, epochs=5, batch_size=256,
                        validation_data=(scaled_test_images, test_labels))
    return history


history = train_model(scaled_train_images, train_labels, scaled_test_images, test_labels)

# Plot the learning curves
# We will now plot two graphs:
# Epoch vs accuracy
# Epoch vs loss
# We will load the model history into a pandas DataFrame and use the plot method to output
# the required graphs.
frame = pd.DataFrame(history.history)
print(frame.columns)

plt = frame.plot(y=["accuracy", "val_accuracy"], color=['b', 'g'], xlabel="Epochs",
                 ylabel=["Train Accuracy", "Validation Accuracy"], title='Accuracy vs Epochs', legend=True)
plt.xaxis.set_major_locator(MaxNLocator(integer=True))

# Run this cell to make the Loss vs Epochs plot
frame2 = pd.DataFrame(history.history)
plt2 = frame2.plot(y=["loss", "val_loss"], color=['b', 'g'], xlabel="Epochs",
                   ylabel=["Train Loss", "Validation Loss"], title='Loss vs Epochs', legend=True)
plt2.xaxis.set_major_locator(MaxNLocator(integer=True))

# The returned history object holds a dictionay record of the loss values and
# metric value(s) during training for each Epoch (in this case length 2)
print(history.history)
print(len(history.history))


# Run your function to evaluate the model
def evaluate_model(scaled_test_images, test_labels):
    test_loss, test_accuracy = model.evaluate(scaled_test_images, test_labels, verbose=2)
    return test_loss, test_accuracy


test_loss, test_accuracy = evaluate_model(scaled_test_images, test_labels)

# loss: 4.88% - accuracy: 98.55%
print('> Test accuracy: %.3f %%' % (test_accuracy * 100.0))
print('> Test loss: %.3f %%' % (test_loss * 100.0))

'''Model predictions Let's see some model predictions! We will randomly select four images from the test data, 
and display the image and label for each. 

For each test image, model's prediction (the label with maximum probability) is shown, together with a plot showing 
the model's categorical distribution. '''
# Run this cell to get model predictions on randomly selected test images

num_test_images = scaled_test_images.shape[0]

random_inx = np.random.choice(num_test_images, 4)
random_test_images = scaled_test_images[random_inx, ...]
random_test_labels = test_labels[random_inx, ...]

predictions = model.predict(random_test_images)

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.4, wspace=-0.2)

for i, (prediction, image, label) in enumerate(zip(predictions, random_test_images, random_test_labels)):
    axes[i, 0].imshow(np.squeeze(image))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(10., -1.5, f'Digit {label}')
    axes[i, 1].bar(np.arange(len(prediction)), prediction)
    axes[i, 1].set_xticks(np.arange(len(prediction)))
    axes[i, 1].set_title(f"Categorical distribution. Model prediction: {np.argmax(prediction)}")

plt.show()


# Improvement to Learning############################################
#####################################################################
####################################################################


# Batch Normalization
# Batch normalization can be used after convolutional and fully connected layers.
# It has the effect of changing the distribution of the output of the layer,
# specifically by standardizing the outputs.
# This has the effect of stabilizing and accelerating the learning process.

# We can update the model definition to use batch normalization after the
# activation function for the convolutional and dense layers of our baseline model. The updated version of
# define_model() function with batch normalization is listed below.

# this function builds the model and compiles in one step (vs what we did early, building 2 functions)
def define_model():
    model = Sequential(
        [Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
         # default data_format='channels_last'
         MaxPooling2D((2, 2)),
         Flatten(),
         Dense(100, activation='relu', kernel_initializer='he_uniform'),
         BatchNormalization(),
         Dense(10, activation='softmax')
         ])
    # compile Model
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model_BatchNorm = define_model()
print(model_BatchNorm.summary())


def train_model(scaled_train_images, train_labels):
    history_BatchNorm = model_BatchNorm.fit(scaled_train_images, train_labels, epochs=5, batch_size=256)
    # save model
    model_BatchNorm.save('final_model.h5')
    return history_BatchNorm


history_BatchNorm = train_model(scaled_train_images, train_labels)


def evaluate_model_bn(scaled_test_images, test_labels):
    test_loss_BN, test_accuracy_BN = model_BatchNorm.evaluate(scaled_test_images, test_labels, verbose=2)
    return test_loss_BN, test_accuracy_BN


test_loss_BN, test_accuracy_BN = evaluate_model_bn(scaled_test_images, test_labels)

# Batch Norm Accuracy 98.58%, Loss 4.29%
print('> Batch Norm Test accuracy: %.3f %%' % (test_accuracy_BN * 100.0))
print('> Batch Norm Test loss: %.3f %%' % (test_loss_BN * 100.0))
