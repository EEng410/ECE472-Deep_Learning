#!/bin/env python

import tensorflow as tf


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, M,  bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable(
            rng.normal(shape=[M, num_outputs], stddev=stddev),
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
    
class BasisExpansion(tf.Module):
    def __init__(self, num_inputs, M):
        rng = tf.random.get_global_generator()
        #Initialize with some stddev
        stddev = rng.uniform(shape = [1, M])
        #Initialize means uniformly distrubuted on [0, 1]
        means = rng.uniform(shape = [1, M])
        self.sigma = tf.Variable(
            stddev,
            trainable = True,
            name = "Basis/sigma"
        )
        self.mu = tf.Variable(
            means,
            trainable = True,
            name = "Basis/mu"
        )
    def __call__(self, x):
        phi = tf.math.exp(-1*(tf.divide(x - tf.repeat(self.mu, num_inputs, axis = 0), tf.repeat(self.sigma, num_inputs, axis = 0)))**2)
        return phi
    


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


if __name__ == "__main__":
    import argparse

    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml

    from tqdm import trange

    import numpy as np

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
    M = 6 # Number of basis functions

    # Data generation
    x = rng.uniform(shape=(num_samples, num_inputs))
    noise = rng.normal(shape = (1, num_outputs), mean = 0.0, stddev = config["data"]["noise_stddev"])
    y = np.sin(2*np.pi*x) + noise

    linear = Linear(num_inputs, num_outputs, M)


    # breakpoint()

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)
    basis = BasisExpansion(num_inputs, M)
    for i in bar:
        # Implement SGD by setting the batch size to 1. then, we only look at gradient and loss at only one random index. 
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.gather(y, batch_indices)

            # breakpoint()

            phis = basis(x_batch)
            # breakpoint()
            y_hat = linear(phis)
            loss = tf.math.reduce_mean((y_batch - y_hat) ** 2)
            # breakpoint()
        # ['Linear/w:0', 'Linear/b:0', 'Basis/sigma:0', 'Basis/mu:0']
        # Implement stochastic gradient descent on w, mu, sigma 
        grads = tape.gradient(loss, [linear.w, linear.b, basis.mu, basis.sigma])
        grad_update(step_size, [linear.w, linear.b, basis.mu, basis.sigma] , grads)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()
    # breakpoint()
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(x.numpy().squeeze(), y.numpy().squeeze(), "x")

    a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 50)[:, tf.newaxis]
    phi = basis(a)
    ax[0].plot(a.numpy().squeeze(), linear(phi).numpy().squeeze(), "-")

    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Linear fit using SGD")
    
    h = ax[0].set_ylabel("y", labelpad=10)
    h.set_rotation(0)
    for i in range(M):
        ax[1].plot(a.numpy().squeeze(), phi[:, i].numpy().squeeze(), "-")
    
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Gaussian Basis Functions")
    fig.savefig("plot.pdf")
