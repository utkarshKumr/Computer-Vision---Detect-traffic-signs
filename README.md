# Computer-Vision---Detect-traffic-signs

Computer Vision CSCI-GA.2272-001 Assignment 1, part 1.
Fall 2021 semester.

Due date: September 30th 2021.

Introduction
This assignment is an introduction to using PyTorch for training simple neural net models. Two different datasets will be used:

MNIST digits [handwritten digits]
CIFAR-10 [32x32 resolution color images of 10 object classes].
Requirements
You should perform this assignment in PyTorch by modifying this ipython notebook (File-->Save a copy...).

To install PyTorch, follow instructions at http://pytorch.org/

Please submit your assignment by uploading this iPython notebook to Brightspace.

Warmup [5%]
It is always good practice to visually inspect your data before trying to train a model, since it lets you check for problems and get a feel for the task at hand.

MNIST is a dataset of 70,000 grayscale hand-written digits (0 through 9). 60,000 of these are training images. 10,000 are a held out test set.

CIFAR-10 is a dataset of 60,000 color images (32 by 32 resolution) across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The train/test split is 50k/10k.

Use matplotlib and ipython notebook's visualization capabilities to display some of these images. See this PyTorch tutorial page for hints on how to achieve this.

** Relevant Cell: "Data Loading" **

Training a Single Layer Network on MNIST [10%]
Start by running the training on MNIST. By default if you run this notebook successfully, it will train on MNIST.

This will initialize a single layer model train it on the 60,000 MNIST training images for 1 epoch (passes through the training data).

The loss function cross_entropy computes a Logarithm of the Softmax on the output of the neural network, and then computes the negative log-likelihood w.r.t. the given target.

The default values for the learning rate, batch size and number of epochs are given in the "options" cell of this notebook. Unless otherwise specified, use the default values throughout this assignment.

Note the decrease in training loss and corresponding decrease in validation errors.

Paste the output into your report. (a): Add code to plot out the network weights as images (one for each output, of size 28 by 28) after the last epoch. Grab a screenshot of the figure and include it in your report. (Hint threads: #1 #2)

(b): Reduce the number of training examples to just 50. [Hint: limit the iterator in the train function]. Paste the output into your report and explain what is happening to the model.

Training a Multi-Layer Network on MNIST [10%]
Add an extra layer to the network with 1000 hidden units and a tanh non-linearity. [Hint: modify the Net class]. Train the model for 10 epochs and save the output into your report.
Now set the learning rate to 10 and observe what happens during training. Save the output in your report and give a brief explanation
Training a Convolutional Network on CIFAR [25%]
To change over to the CIFAR-10 dataset, change the options cell's dataset variable to 'cifar10'.

Create a convolutional network with the following architecture:
Convolution with 5 by 5 filters, 16 feature maps + Tanh nonlinearity.
2 by 2 max pooling (non-overlapping).
Convolution with 5 by 5 filters, 128 feature maps + Tanh nonlinearity.
2 by 2 max pooling (non-overlapping).
Flatten to vector.
Linear layer with 64 hidden units + Tanh nonlinearity.
Linear layer to 10 output units.
Train it for 20 epochs on the CIFAR-10 training set and copy the output into your report, along with a image of the first layer filters.

Hints: Follow the first PyTorch tutorial or look at the MNIST example. Also, you may find training is faster if you use a GPU runtime (RunTime-->Change Runtime Type-->GPU).

Give a breakdown of the parameters within the above model, and the overall number.
