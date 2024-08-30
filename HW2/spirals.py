#!/bin/env python

import tensorflow as tf
from sklearn.inspection import DecisionBoundaryDisplay

class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

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

class MLP(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, hidden_activation = tf.identity, output_activation = tf.identity, bias=True):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        rng = tf.random.get_global_generator()
        self.bias = bias
        # weights were getting too large so I squashed them down by adding hidden_layer_width
        # stddev = tf.math.sqrt(2 / (hidden_layer_width))

        self.input_layer = Linear(num_inputs, hidden_layer_width)

        # self.w = tf.Variable(
        #     rng.normal(shape=[hidden_layer_width, hidden_layer_width, num_hidden_layers-1], stddev=stddev),
        #     trainable=True,
        #     name="MLP/w",
        # )

        self.hidden_layers = [Linear(hidden_layer_width, hidden_layer_width) for i in range(num_hidden_layers)]

        self.output_layer = Linear(hidden_layer_width, num_outputs)

        # if self.bias:
        #     self.b = tf.Variable(
        #         tf.zeros(
        #             shape=[1, num_hidden_layers],
        #         ),
        #         trainable=True,
        #         name="MLP/b",
        #     )

    def __call__(self, x):
        z = tf.cast(x, 'float32')
        z = tf.transpose(z)
        # z = tf.reshape(z, [-1, 2])
        # Iterate through the hidden layers
        # breakpoint()
        z = self.input_layer(z)
        # z = tf.add(z, tf.slice(self.b, [0, 0], [1, 1]))
        
        for layer in range(self.num_hidden_layers):
            # breakpoint()
            # z = tf.tensordot(z, tf.slice(self.w, [0, 0, layer+1], [self.hidden_layer_width, self.hidden_layer_width, 1]), axes = ((0), (1))) 
            
            # z =  z @ tf.slice(self.w, [0, 0, layer], [self.hidden_layer_width, self.hidden_layer_width, 1])
            # z = tf.transpose(z)
            # if self.bias:
            #     # breakpoint()
            #     z = tf.add(z, tf.slice(self.b, [0, layer+1], [1, 1]))
            # z = self.hidden_activation(z)
            z = self.hidden_activation(self.hidden_layers[layer](z))
        # ouptut layer
        # breakpoint()
        p = self.output_activation(self.output_layer(z))

        # if p >= 0.5:
        #     return 1
        # if p < 0.5:
        #     return 0
        return p

def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        # breakpoint()
        var.assign_sub(step_size * grad)



if __name__ == "__main__":
    import argparse
    import numpy as np
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml

    from tqdm import trange

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
    num_inputs = 2
    num_outputs = 1

    random = np.random.default_rng()
    theta = random.uniform(np.pi/2, 6*np.pi, int(num_samples/2))
    r = theta
    var =  config["data"]["noise_stddev"]
    noise = random.normal(0, var, [2, num_samples])

    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)

    x1 = -r * np.cos(theta)
    y1 = -r * np.sin(theta)

    coords = np.array([np.concatenate((x0, x1)), np.concatenate((y0, y1))])
    coords = coords + noise
    # breakpoint()
    inputs = coords
    output_classes = np.zeros([1, num_samples])
    midpoint = int(output_classes.shape[1]/2)
    output_classes = np.vstack((np.zeros([1, int(num_samples/2)]), np.ones([1, int(num_samples/2)]))).reshape([1, -1])
    # linear = Linear(num_inputs, num_outputs)
    

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    refresh_rate = config["display"]["refresh_rate"]
    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, tf.nn.relu, tf.nn.sigmoid)
    bar = trange(num_iters)
    # Initialize 1st and 2nd moment vectors
    m = [tf.zeros(mlp.trainable_variables[i].shape, dtype = 'float32') for i in range(len(mlp.trainable_variables))]
    v = [tf.zeros(mlp.trainable_variables[i].shape, dtype = 'float32') for i in range(len(mlp.trainable_variables))]
    # Unless the weighting for the L2 penalty is this small, the algorithm cannot converge. It will instead find that loss is minimized by setting all weights to 0 and guessing  0 every time
    gamma = 0.001
    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            # tape.watch(mlp.b)
            # x_batch = tf.gather(inputs[0, :], batch_indices)
            # y_batch = tf.gather(inputs[1, :], batch_indices)
            x_batch = tf.gather(inputs, batch_indices, axis=1)
            y_batch = tf.gather(output_classes, batch_indices, axis = 1)
            y_batch = tf.cast(y_batch, 'float32')

            y_hat = []
            # for k in range(tf.size(y_batch)):
            #     # breakpoint()
            #     # y_hat[0, i] = mlp(x_batch[:, i])
            #     # breakpoint()
            #     y_hat.append(mlp(x_batch[:, k]))
            # # breakpoint()
            # y_hat = tf.stack(y_hat)
            y_hat = mlp(x_batch)
            # breakpoint()
            # loss = tf.norm((tf.reshape(mlp.w, [1, -1])), ord = 'euclidean') - 1/batch_size * (tf.tensordot(y_batch, tf.math.log(y_hat+(1e-7)), axes = ((1), (0)))+ tf.tensordot(1-y_batch, tf.math.log(1-y_hat+(1e-7)), axes = ((1), (0))))
            loss =  - 1/batch_size * (tf.tensordot(y_batch, tf.math.log(y_hat+(1e-7)), axes = ((1), (0)))+ tf.tensordot(1-y_batch, tf.math.log(1-y_hat+(1e-7)), axes = ((1), (0))))
            # add L2 regularization
            for var in mlp.trainable_variables:
                loss += gamma* tf.math.reduce_sum(tf.multiply(tf.reshape(var, [1, -1]), tf.reshape(var, [1, -1])))

        # print(loss)
        grads = tape.gradient(loss, mlp.trainable_variables)
        # breakpoint()

        # Implementing Adam
        beta_1 = 0.9
        beta_2 = 0.999
        # breakpoint()
        m = tuple(map(sum, zip(tuple([(beta_1)*m_i for m_i in m]), tuple([(1-beta_1)*grad for grad in grads]))))
        grads_sq = [tf.math.multiply(grad, grad) for grad in grads]
        v = tuple(map(sum, zip(tuple([(beta_2)*v_i for v_i in v]), tuple([(1-beta_2)*grad for grad in grads_sq]))))
        
        m_hat = tuple([1/(1-beta_1**(i+1))*m_i for m_i in m])
        v_hat = tuple([1/(1-beta_2**(i+1))*v_i for v_i in v])

        grad_new = [tf.math.divide(aItem, tf.math.sqrt(bItem)+(1e-8)) for aItem, bItem in zip(m_hat, v_hat)]
        # breakpoint()
        grad_update(step_size, mlp.trainable_variables, grad_new)
        
        # grad_update(step_size, mlp.trainable_variables, grads)
        # step_size *= decay_rate
        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy().squeeze():0.4f}, step_size => {step_size:0.4f}"
            )
            
            bar.refresh()

    fig, ax = plt.subplots()
    
   
    x, y = np.meshgrid(np.linspace(-25.,25., 600), np.linspace(-25.,25.,600))
    grid = np.vstack([x.ravel(), y.ravel()])
    # Reshape the x, y grid to fit a 2 x [-1] shape
    y_pred = np.reshape(mlp(grid), x.shape)
    display = DecisionBoundaryDisplay(
        xx0=x, xx1=y, response=y_pred
    )
    display.plot()

    display.ax_.scatter(coords[0, :], coords[1, :], c = output_classes, edgecolors='k')

    display.figure_.savefig("plot.pdf")