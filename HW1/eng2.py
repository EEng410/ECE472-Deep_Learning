import tensorflow as tf


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

    # x = rng.uniform(shape=(num_samples, num_inputs))
    # w = rng.normal(shape=(num_inputs, num_outputs))
    # b = rng.normal(shape=(1, num_outputs))
    # y = rng.normal(
    #     shape=(num_samples, num_outputs),
    #     mean=x @ w + b,
    #     stddev=config["data"]["noise_stddev"],
    # )

    generator = np.random.Generator()

    linear = Linear(num_inputs, num_outputs)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.gather(y, batch_indices)

            y_hat = linear(x_batch)
            loss = tf.math.reduce_mean((y_batch - y_hat) ** 2)

        grads = tape.gradient(loss, linear.trainable_variables)
        grad_update(step_size, linear.trainable_variables, grads)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

    fig, ax = plt.subplots()

    ax.plot(x.numpy().squeeze(), y.numpy().squeeze(), "x")

    a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 100)[:, tf.newaxis]
    ax.plot(a.numpy().squeeze(), linear(a).numpy().squeeze(), "-")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear fit using SGD")
    
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    fig.savefig("plot.pdf")