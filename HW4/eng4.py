#!/bin/env python

from typing import Any
import tensorflow as tf


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.cast(tf.math.sqrt(2 / (num_inputs + num_outputs)), dtype = tf.float32)

        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z

class GroupNorm(tf.Module):
    def __init__(self, G, eps=1e-5):
        # The following implementation was taken from the GroupNorm paper. 
        # x: input features with shape [N,C,H,W]
        # gamma, beta: scale and offset, with shape [1,C,1,1]
        # G: number of groups for GN
        self.gamma = tf.Variable(
            1.0, 
            trainable = True,
            name = "GN/gamma"
        )
        self.beta = tf.Variable(
            0.0, 
            trainable = True,
            name = "GN/gamma"
        )
        self.G = G
        self.eps = eps
    def __call__(self, x):
        
        N, H, W, C = x.shape
        breakpoint()
        x = tf.reshape(x, [N, self.G, C // self.G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta

    

class Conv2d(tf.Module):
    def __init__(self, k, in_channels, out_channels, stride = 1, dropout_prob = 0.2, activation = tf.identity):
        self.dropout_prob = dropout_prob
        self.k = k
        stddev = tf.cast(tf.math.sqrt(1 / k), dtype = tf.float32)
        self.stride = stride
        self.activation = activation
        self.kernel = tf.Variable(
            #[filter_height, filter_width, in_channels, out_channels]
            # tf.ones(shape=[self.k, self.k, in_channels, out_channels]),
            rng.normal(shape=[self.k, self.k, in_channels, out_channels], stddev= stddev),
            trainable=True,
            name="Conv2d/kernel",
        )

    def __call__(self, x):
        z = tf.nn.conv2d(x, self.kernel, strides=[1, self.stride, self.stride, 1], padding = "SAME")
        # apply the activation function
        z = self.activation(z)
        # # apply dropout
        rng = tf.random.get_global_generator()
        dropout_vec = tf.cast(tf.where(rng.uniform(shape = z.shape, minval = 0, maxval = 1) < (1-self.dropout_prob), 1, 0), tf.float32)
        z = tf.math.multiply(z, dropout_vec) / (1-self.dropout_prob)
        return z
    
# Pooling class not used, but I probably should have. 
class AvgPooling(tf.Module):
    def __init__(self, k, in_channels, out_channels):
        self.k = k
        # make an averaging kernel in the pooling layer, do not make it trainable
        self.kernel = tf.Variable(
            tf.ones(shape=[self.k, self.k, in_channels, out_channels])/k**2,
            # trainable=True,
            name="AvgPooling/kernel",
        )

    def __call__(self, x):
        # perform the convolution on the batch
        z = tf.nn.conv2d(x, self.kernel, strides=[1, self.k, self.k, 1], padding = "SAME")
        return z

class ResidualBlock(tf.Module):
    def __init__(self, k, in_channels, out_channels, G, residual_activation = tf.nn.relu, stride = 1, dropout_prob = 0.2, conv_activation = tf.identity, awgn = True, dropout = True, layers = 2):
        #k, in_channels, out_channels should be lists of integers
        self.residual_activation = residual_activation
        self.layers = layers
        self.group_norm_layers = [GroupNorm(G) for i in range(layers)]
        self.conv_layers = [Conv2d(k[i], in_channels, out_channels, stride, dropout_prob, activation = conv_activation) for i in range(layers)]
    def __call__(self, x):
        carry_out = x
        for i in range(self.layers):
            x = self.group_norm_layers[i](x)
            x = self.residual_activation(x)
            x = self.conv_layers[i](x)
        output = carry_out + x
        return output


class Classifier(tf.Module):
    def __init__(self, input_size, input_depth, layer_depths, layer_kernel_sizes, num_classes, num_residual_layers, strides = None, G = 4, residual_activation = tf.nn.relu, stride = 1, dropout_prob = 0.2, conv_activation = tf.identity, awgn = True, dropout = True, layers = 2):
        if strides is None: strides = tf.ones(shape = [num_residual_layers])

        #should probably throw an error if num_conv_layers doesn't match length of layer_depths or layer_kernel_sizes

        self.input_depth = input_depth
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.num_classes = num_classes
        # def __init__(self, k, in_channels, out_channels, stride = 1, dropout_prob = 0.1, activation = tf.identity):
        # breakpoint()
        self.num_residual_layers = num_residual_layers
        self.residual_layers = []
        for i in range(num_residual_layers):
            if i == 0:
                self.residual_layers.append(Conv2d(layer_kernel_sizes[i], input_depth, layer_depths[i], dropout_prob=dropout_prob, activation=conv_activation))
            else:
                self.residual_layers.append(ResidualBlock(layer_kernel_sizes[i], layer_depths[i-1], layer_depths[i], G, residual_activation, stride, dropout_prob, conv_activation, awgn, dropout, layers))
        self.output_layer = Linear(input_size*layer_depths[-1], self.num_classes)
        # self.output_layer = Linear(num_classes)
    def __call__(self, x):
        # conv_layers
        conv_output = tf.reshape(x, [x.shape[0], x.shape[1], x.shape[2], self.input_depth])
        for i in range(self.num_residual_layers):
            # breakpoint()
            conv_output = self.residual_layers[i](conv_output)

        #pooling
        # flatten conv output
        # breakpoint()
        out_flat = tf.reshape(conv_output, [x.shape[0], -1])
        # get length of flattened output
        # output_len = tf.cast(tf.size(out_flat)/x.shape[0], tf.int64)
        # breakpoint()

        
        # out_flat = tf.reshape(out_flat, [-1, out_flat.shape[-1]])
        z = self.output_layer(out_flat)
        z = tf.nn.softmax(z)
        return z

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)

if __name__ == "__main__":
    import argparse

    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml
    import numpy as np

    from tqdm import trange
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    num_samples = config["data"]["num_samples"]
    num_inputs = 1
    num_outputs = 1

    enc = OneHotEncoder(handle_unknown="ignore")

    data = unpickle("./cifar-10-batches-py/data_batch_1")

    images = data[b'data']
    labels = data[b'labels']
    images.shape
    images = np.reshape(images, (10000, 1024, 3))

    images = np.reshape(images, (10000, 32, 32, 3)).astype("float32")
    # format is list of arrays, convert to array first and then to tensor

    # images_trainval = np.array(images_trainval)
    # images_test = np.array(images_test)

    # labels_trainval = np.array(labels_trainval)
    # labels_test = np.array(labels_test)    


    # train/val split
    # arrays_train, arrays_val, labels_train, labels_val = train_test_split(images_trainval, labels_trainval, test_size = 0.3, random_state = 42)

    # breakpoint()

    # images_train = tf.convert_to_tensor(arrays_train.reshape(-1, 28, 28), dtype = tf.float32)
    # images_val = tf.convert_to_tensor(arrays_val.reshape(-1, 28, 28), dtype = tf.float32)
    # images_test = tf.convert_to_tensor(images_test.reshape(-1, 28, 28), dtype = tf.float32)

    # # One-hot encoding to make compatible with 
    # labels_train = tf.one_hot(labels_train, 10)
    # labels_val = tf.one_hot(labels_val, 10)
    # labels_test = tf.one_hot(labels_test, 10)

    # def __init__(self, input_size, input_depth, layer_depths, layer_kernel_sizes, num_classes, num_residual_layers, strides = None, G = 32, residual_activation = tf.nn.relu, stride = 1, dropout_prob = 0.2, conv_activation = tf.identity, awgn = True, dropout = True, layers = 2):
    classifier = Classifier(1024, 3, [12, 12, 12], [[3, 3], [2, 2], [2, 2]], 10, 3, dropout_prob= 0.1)

    # breakpoint()
    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    gamma = config["learning"]["weight_decay"]
    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)
    cce = tf.keras.losses.CategoricalCrossentropy()
    beta_1 = 0.9
    beta_2 = 0.999
    # Initialize 1st and 2nd moment vectors
    m = [tf.zeros(classifier.trainable_variables[i].shape, dtype = 'float32') for i in range(len(classifier.trainable_variables))]
    v = [tf.zeros(classifier.trainable_variables[i].shape, dtype = 'float32') for i in range(len(classifier.trainable_variables))]
    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )

        with tf.GradientTape() as tape:
            x_batch = tf.gather(images, batch_indices)
            y_batch = tf.gather(labels, batch_indices)

            y_hat = classifier(x_batch)
            loss = cce(y_batch, y_hat)
            # for var in classifier.trainable_variables:
            #     loss += gamma * tf.math.reduce_sum(tf.multiply(tf.reshape(var, [1, -1]), tf.reshape(var, [1, -1])))
        grads = tape.gradient(loss, classifier.trainable_variables)
        # breakpoint()
        #Implementing Adam
        
        # w = gamma/step_size
        # breakpoint()
        m = tuple(map(sum, zip(tuple([(beta_1)*m_i for m_i in m]), tuple([(1-beta_1)*grad for grad in grads]))))
        grads_sq = [tf.math.multiply(grad, grad) for grad in grads]
        v = tuple(map(sum, zip(tuple([(beta_2)*v_i for v_i in v]), tuple([(1-beta_2)*grad for grad in grads_sq]))))
        
        m_hat = tuple([1/(1-beta_1**(i+1))*m_i for m_i in m])
        v_hat = tuple([1/(1-beta_2**(i+1))*v_i for v_i in v])

        # Implement AdamW
        grad_new = [tf.math.divide(aItem, tf.math.sqrt(bItem)+(1e-8)) + gamma * cItem for aItem, bItem, cItem in zip(m_hat, v_hat, classifier.trainable_variables)]
        # grad_new = [tf.math.divide(aItem, tf.math.sqrt(bItem)+(1e-8)) for aItem, bItem in zip(m_hat, v_hat)]
        # breakpoint()
        grad_update(step_size, classifier.trainable_variables, grad_new)

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()
    # breakpoint()

    # Accuracy calculation (one-hot labels, input set, classifier)
    def accuracy(truth, set, classifier):
        preds = tf.argmax(classifier(set), axis = 1)
        truth = tf.argmax(truth, axis = 1)
        accuracy = tf.math.count_nonzero(tf.equal(truth,preds)) / preds.shape[0] * 100
        return accuracy

    # on validation data

    # validation accuracy 95.56%
    print(accuracy(labels_val, images_val, classifier))
    
    # train accuracy 96.8%
    print(accuracy(labels_train, images_train, classifier))

    # set a breakpoint so I can look at train/val accuracy without looking at test accuracy
    # breakpoint()

    # test accuracy 95.91%
    print(accuracy(labels_test, images_test, classifier))

