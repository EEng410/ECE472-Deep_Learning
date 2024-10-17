#!/bin/env python
from typing import Any
import tensorflow as tf
from tensorflow.keras import layers

class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()
        num_inputs = tf.cast(num_inputs, tf.int32)

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
    def __init__(self, G, C, eps=1e-5):
        # The following implementation was taken from the GroupNorm paper. 
        # x: input features with shape [N,C,H,W]
        # gamma, beta: scale and offset, with shape [1,C,1,1]
        # G: number of groups for GN
        self.gamma = tf.Variable(
            tf.ones(shape = [1, C, 1, 1], dtype = tf.float32), 
            trainable = True,
            name = "GN/gamma"
        )
        self.beta = tf.Variable(
            tf.zeros(shape = [1, C, 1, 1], dtype = tf.float32), 
            trainable = True,
            name = "GN/beta"
        )
        self.G = G
        self.eps = eps
    def __call__(self, x):
        
        N, H, W, C = x.shape
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, [N, self.G, C // self.G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, [N, C, H, W])
        x = x * self.gamma + self.beta
        x = tf.transpose(x, [0, 2, 3, 1])
        return x

    

class Conv2d(tf.Module):
    def __init__(self, k, in_channels, out_channels, dropout_prob, stride = 1, activation = tf.nn.relu):
        self.dropout_prob = dropout_prob
        self.k = k
        # use the proper initialization scheme proposed in the PReLU paper. Insane how much of a difference this makes.
        stddev = tf.cast(tf.math.sqrt(2 / (32**2 * in_channels)), dtype = tf.float32)
        self.stride = stride
        self.activation = activation
        self.kernel = tf.Variable(
            #[filter_height, filter_width, in_channels, out_channels]
            # tf.ones(shape=[self.k, self.k, in_channels, out_channels]),
            rng.normal(shape=[self.k, self.k, in_channels, out_channels], stddev= stddev),
            trainable=True,
            name="Conv2d/kernel",
        )

    def __call__(self, x, inference):
        z = tf.nn.conv2d(x, self.kernel, strides=[1, self.stride, self.stride, 1], padding = "SAME")
        # apply the activation function
        z = self.activation(z)
        # # apply dropout except for at inference time
        if inference == False:
            # rng = tf.random.get_global_generator()
            # dropout_vec = tf.cast(tf.where(rng.uniform(shape = z.shape, minval = 0, maxval = 1) < (1-self.dropout_prob), 1, 0), tf.float32)
            # z = tf.math.multiply(z, dropout_vec) / (1-self.dropout_prob)
            z = tf.nn.dropout(z, self.dropout_prob)
        return z
    
# Pooling class not used, but I probably should have. 
class AvgPooling(tf.Module):
    def __init__(self, k, in_channels, out_channels):
        self.k = k
        # make an averaging kernel in the pooling layer, do not make it trainable
        self.kernel = tf.Variable(
            tf.ones(shape=[self.k, self.k, in_channels, out_channels])/(k**2),
            # trainable=True,
            name="AvgPooling/kernel",
        )

    def __call__(self, x):
        # perform the convolution on the batch
        z = tf.nn.conv2d(x, self.kernel, strides=[1, self.k, self.k, 1], padding = "SAME")
        return z

class ResidualBlock(tf.Module):
    def __init__(self, k, in_channels, out_channels, G, input_size, num_layers, residual_activation = tf.nn.relu, stride = 1, dropout_prob = 0.2, conv_activation = tf.nn.relu, awgn = True, dropout = True):
        #k, in_channels, out_channels should be lists of integers
        self.residual_activation = residual_activation
        self.layers = num_layers

        self.group_norm_layers = [GroupNorm(G, out_channels) for i in range(num_layers)]
        self.conv_layers = [Conv2d(k[i], in_channels, out_channels, stride = stride, dropout_prob = dropout_prob, activation = conv_activation) for i in range(num_layers)]
        
    def __call__(self, x, inference = False):
        carry_out = x
        for i in range(self.layers):
            x = self.conv_layers[i](x, inference=inference)
            x = self.group_norm_layers[i](x)
            x = self.residual_activation(x)           
        output = carry_out + x
        if inference == False and self.awgn == True:
            rng = tf.random.get_global_generator()
            awgn_vec = rng.normal(shape = output.shape, stddev = 0.05)
            output = output+awgn_vec
        return output


class Classifier(tf.Module):
    def __init__(self, input_size, input_depth, layer_depths, layer_kernel_sizes, num_classes, num_residual_layers, strides = None, G = 4, residual_activation = tf.nn.relu, stride = 1, dropout_prob = 0.2, conv_activation = tf.nn.relu, awgn = True, dropout = True, num_layers = 1, pooling_factor = 4):
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
                self.residual_layers.append(Conv2d(layer_kernel_sizes[i][0], input_depth, layer_depths[i], dropout_prob=dropout_prob, activation=conv_activation))
            else:
                self.residual_layers.append(ResidualBlock(layer_kernel_sizes[i], layer_depths[i-1], layer_depths[i], G, input_size, num_layers, residual_activation, stride, dropout_prob, conv_activation, awgn, dropout))
        self.output_layer = Linear(input_size*layer_depths[-1]/(pooling_factor**2), self.num_classes)
        self.residual_layers.append(ResidualBlock(layer_kernel_sizes[i], layer_depths[i-1], layer_depths[i], G, residual_activation, stride, dropout_prob, conv_activation, awgn, dropout, num_layers))
        self.output_layer = Linear(input_size*layer_depths[-1], self.num_classes)
        # self.output_layer = Linear(num_classes)

        self.data_augmentation_layer = tf.keras.models.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
        ])
        # self.pooling_layer = AvgPooling(pooling_factor, layer_depths[-1], layer_depths[-1])
    def __call__(self, x, inference = False):
        # conv_layers
        # if inference == False:
            # x = self.data_augmentation_layer(x)
        conv_output = tf.reshape(x, [x.shape[0], x.shape[1], x.shape[2], self.input_depth])
        for i in range(self.num_residual_layers):
            # breakpoint()
            conv_output = self.residual_layers[i](conv_output, inference)
        # conv_output = self.pooling_layer(conv_output)
        #pooling
        # flatten conv output
        # breakpoint()
        out_flat = tf.reshape(conv_output, [x.shape[0], -1])
        # get length of flattened output
        # output_len = tf.cast(tf.size(out_flat)/x.shape[0], tf.int64)
        # breakpoint()

        
        # out_flat = tf.reshape(out_flat, [-1, out_flat.shape[-1]])
        # breakpoint()
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

    data_1 = unpickle("./cifar-10-batches-py/data_batch_1")
    data_2 = unpickle("./cifar-10-batches-py/data_batch_2")
    data_3 = unpickle("./cifar-10-batches-py/data_batch_3")
    data_4 = unpickle("./cifar-10-batches-py/data_batch_4")
    data_5 = unpickle("./cifar-10-batches-py/data_batch_5")

    images = np.concatenate((data_1[b'data'], data_2[b'data'], data_3[b'data'], data_4[b'data'], data_5[b'data']), axis = 0)
    labels = np.concatenate((data_1[b'labels'], data_2[b'labels'], data_3[b'labels'], data_4[b'labels'], data_5[b'labels']), axis = 0)

    images = np.reshape(images, (50000, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).astype("float32")
    labels = tf.one_hot(labels, 10)

    data_test = unpickle("./cifar-10-batches-py/test_batch")

    images_test = data_test[b'data']
    labels_test = data_test[b'labels']

    images_test = np.reshape(images_test, (10000, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).astype("float32")
    labels_test = tf.one_hot(labels_test, 10)
    # breakpoint()

    # classifier = unpickle("restarting.pkl")


    # breakpoint()
    # def __init__(self, input_size, input_depth, layer_depths, layer_kernel_sizes, num_classes, num_residual_layers, strides = None, G = 32, residual_activation = tf.nn.relu, stride = 1, dropout_prob = 0.2, conv_activation = tf.identity, awgn = True, dropout = True, layers = 2):
    classifier = Classifier(1024, 3, [32, 64, 128, 256], [[7, 7], [3, 3], [3, 3], [3, 3]], 10, 4, dropout_prob= 0.1)
    classifier = Classifier(1024, 3, [32, 32, 32, 32], [[3, 3], [3, 3], [3, 3], [3, 3]], 10, 4, dropout_prob= 0.1)

    # breakpoint()
    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    gamma = config["learning"]["weight_decay"]
    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)
    cce = tf.keras.losses.CategoricalCrossentropy()
    # beta_1 = 0.9
    # beta_2 = 0.999
    # Initialize 1st and 2nd moment vectors
    # m = [tf.zeros(classifier.trainable_variables[i].shape, dtype = 'float32') for i in range(len(classifier.trainable_variables))]
    # v = [tf.zeros(classifier.trainable_variables[i].shape, dtype = 'float32') for i in range(len(classifier.trainable_variables))]
    # step_size_reduction = True
    # step_size_reduction_2 = True
    loss_vec = np.zeros((num_iters, 1))

    adam = tf.keras.optimizers.AdamW(learning_rate = step_size, weight_decay = gamma)
    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        
        with tf.GradientTape() as tape:
            x_batch = tf.gather(images, batch_indices, axis = 0)
            y_batch = tf.gather(labels, batch_indices, axis = 0)
            # breakpoint()

            y_hat = classifier(x_batch)
            loss = cce(y_batch, y_hat)
            for var in classifier.trainable_variables:
                loss += gamma * tf.math.reduce_sum(tf.multiply(tf.reshape(var, [1, -1]), tf.reshape(var, [1, -1])))
        if step_size_reduction and loss < 3.0:
            step_size = step_size/5
            step_size_reduction = False
        if step_size_reduction_2 and loss < 1.0:
            step_size = step_size/2
            step_size_reduction_2 = False
        grads = tape.gradient(loss, classifier.trainable_variables)
        # breakpoint()
        adam.apply(grads, classifier.trainable_variables)
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
        loss_vec[i] = loss
        # step_size = step_size*decay_rate
    # breakpoint()

    # Accuracy calculation (one-hot labels, input set, classifier)
    def accuracy(truth, set, classifier):
        preds = tf.argmax(classifier(set, inference = True), axis = 1)
        truth = tf.argmax(truth, axis = 1)
        accuracy = tf.math.count_nonzero(tf.equal(truth,preds)) / preds.shape[0] * 100
        return accuracy
    breakpoint()
    print(accuracy(labels_test[0:4999], images_test[0:4999, :, :, :], classifier))
    print(accuracy(labels_test[5000:9999], images_test[5000:9999, :, :, :], classifier))

    plt.figure(figsize=(10, 6))  # Adjust the width and height as desired

    plt.plot(loss_vec)
    plt.savefig("pain.pdf")

