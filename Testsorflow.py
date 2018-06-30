""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
This example is using TensorFlow layers API, see 'convolutional_network_raw' 
example for a raw implementation with variables.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf
import numpy as np

# Training Parameters
learning_rate = 0.01
num_epochs = 1
batch_size = 18

# Network Parameters
num_classes = 100 # QUICKDRAW total classes (1-100 weas)
dropout = 0.25 # Dropout, probability to drop a unit

def buildConv2DLayer(input, out_size,is_training):
        conv = tf.layers.conv2d(input,out_size,3,
                padding = "SAME", activation = tf.nn.relu,
                kernel_initializer = tf.contrib.layers.xavier_initializer())

        return tf.layers.batch_normalization(conv,training = is_training, fused = True)

def buildMaxPool(input):
    return tf.layers.max_pooling2d(input, 3, 2, padding = "SAME")

# Create the neural network
def conv_net(x_dict, n_classes, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x_r = tf.reshape(x, shape=[-1, 128, 128, 1])
        x_c = tf.cast(x_r,tf.float32)

        #resized = tf.image.resize_images(x,[128,128],method=tf.image.ResizeMethod.BILINEAR,align_corners=True)
        print(x_c.graph)
        conv1_1 = buildConv2DLayer(x_c,64,is_training)
        conv1_2 = buildConv2DLayer(conv1_1,64,is_training)
        maxpool1 = buildMaxPool(conv1_2)

        conv2_1 = buildConv2DLayer(maxpool1,128,is_training)
        conv2_2 = buildConv2DLayer(conv2_1,128,is_training)
        maxpool2 = buildMaxPool(conv2_2)

        conv3_1 = buildConv2DLayer(maxpool2,128,is_training)
        conv3_2 = buildConv2DLayer(conv3_1,128,is_training)
        maxpool3 = buildMaxPool(conv3_2)

        conv4_1 = buildConv2DLayer(maxpool3,256,is_training)
        conv4_2 = buildConv2DLayer(conv4_1,256,is_training)
        maxpool4 = buildMaxPool(conv4_2)

        flt = tf.contrib.layers.flatten(maxpool4)

        fc_1 = tf.layers.dense(flt,1024,activation = tf.nn.relu,
                kernel_initializer = tf.contrib.layers.xavier_initializer())
        fc_2 = tf.layers.dense(fc_1,100,
                kernel_initializer = tf.contrib.layers.xavier_initializer())

    return fc_2


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model00
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    print("Accuracy: ",acc_op)
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
config = tf.estimator.RunConfig(
    save_checkpoints_steps = 1000,
    keep_checkpoint_max = 10,
)

model = tf.estimator.Estimator(model_fn, model_dir = "models/network1", config = config)

# Define the input function for training

def parse_function(filename, label):
    img = np.load("data/"+filename.decode())
    return img,label

"""
train_data = np.load("traindata.npy").astype(np.float32)
train_label = np.load("trainlabel.npy").astype(int)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': train_data}, y=train_label,
    batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
# Train the Model
"""

def train_input_fn():

    train_files = [s[:-1] + str(k).zfill(4) + ".npy" for s \
        in open("classes.txt","r").readlines() for k in range(1000)]
    train_labels = [k//1000 for k in range(10**5)]
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files,train_labels))
    #train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.map(
        lambda filename, label: tuple(tf.py_func(
            parse_function, [filename, label], [tf.float64, label.dtype])))
    train_dataset = train_dataset.shuffle(10**5)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat(num_epochs)
    iterator = train_dataset.make_one_shot_iterator() 
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels 
model.train(train_input_fn)

def test_input_fn():

    test_files = [s[:-1] + str(k).zfill(4) + ".npy" for s \
        in open("classes.txt","r").readlines() for k in range(1000.1050)]
    test_labels = [k//50 for k in range(5000)]
    test_dataset = tf.data.Dataset.from_tensor_slices((test_files,test_labels))
    #test_dataset = test_dataset.map(parse_function)
    test_dataset = test_dataset.map(
        lambda filename, label: tuple(tf.py_func(
            parse_function, [filename, label], [tf.float64, label.dtype])))
    
    test_dataset = test_dataset.batch(batch_size)
    iterator = test_dataset.make_one_shot_iterator() 
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels 


# Use the Estimator 'evaluate' method
e = model.evaluate(test_input_fn)

print("Testing Accuracy:", e['accuracy'])
