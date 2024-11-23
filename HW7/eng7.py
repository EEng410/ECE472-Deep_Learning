import tensorflow as tf
from sklearn.inspection import DecisionBoundaryDisplay


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        self.w = tf.Variable(
            rng.uniform(shape=[num_inputs, num_outputs], minval = -1/num_inputs, maxval = 1/num_inputs),
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
    
class SineLayer(tf.Module):
    def __init__(self, num_inputs, num_outputs, first_layer = False, bias=True, omega_0 = 30):
        rng = tf.random.get_global_generator()
        self.omega_0 = omega_0
        self.first_layer = first_layer
        if self.first_layer:
            self.w = tf.Variable(
                rng.uniform(shape=[num_inputs, num_outputs], minval = -1/num_inputs, maxval = 1/num_inputs),
                trainable=True,
                name="SineLayer/w",
            )
        else:
            self.w = tf.Variable(
                rng.uniform(shape=[num_inputs, num_outputs], minval = -tf.math.sqrt(6/num_inputs)/self.omega_0 , maxval = tf.math.sqrt(6/num_inputs)/self.omega_0),
                trainable=True,
                name="SineLayer/w",
            )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="SineLayer/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b
        z = z * self.omega_0
        z = tf.math.sin(z)
        return z

class SIREN(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, output_activation = tf.nn.sigmoid, bias=True): # ReLU just to clip negative outputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.output_activation = output_activation

        self.bias = bias
        self.input_layer = SineLayer(num_inputs, hidden_layer_width, first_layer=True)

        self.hidden_layers = [SineLayer(hidden_layer_width, hidden_layer_width) for i in range(num_hidden_layers)]

        self.output_layer = Linear(hidden_layer_width, num_outputs)


    def __call__(self, x):
        z = tf.cast(x, 'float32')
        z = tf.transpose(z)

        z = self.input_layer(z)
       
        for layer in range(self.num_hidden_layers):
            z = self.hidden_layers[layer](z)
        p = self.output_activation(self.output_layer(z))
        return p
    
if __name__ == "__main__":
    import argparse
    import numpy as np
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml

    from tqdm import trange
    from PIL import Image

    num_iters = 5000
    step_size = 1e-4
    decay_rate = 0.999
    gamma = 0.0004
    num_hidden_layers = 6
    hidden_layer_width = 256
    refresh_rate = 10

    optimizer = tf.keras.optimizers.Adam(learning_rate = step_size)
    bar = trange(num_iters)
    # Grabbed image from Wikipedia
    img = Image.open('HW7/Testcard_F.jpg')

    img_tensor = tf.convert_to_tensor(np.asarray(img))

    # Normalize Tensor for training
    img_tensor = img_tensor/tf.math.reduce_max(img_tensor)

    # Generate the coordinate 
    X, Y = tf.meshgrid(np.linspace(-1, 1, img_tensor.shape[1]), np.linspace(-1, 1, img_tensor.shape[0]))
    coord_tensor = tf.transpose(tf.squeeze(tf.Variable([X, Y])))
    coord_tensor = tf.reshape(coord_tensor, shape = [2, -1])

    # Flatten image tensor
    img_tensor = tf.reshape(img_tensor, shape = [-1, 3])

    img_size = img_tensor.shape[0]



    # Initialize SIREN Model
    siren = SIREN(2, 3, num_hidden_layers, hidden_layer_width)
    bar = trange(num_iters)
    for i in bar:
        x_batch = coord_tensor
        y_batch = img_tensor
        with tf.GradientTape(persistent = True) as tape:
            y_hat = siren(x_batch)
            loss =  tf.math.reduce_mean((y_batch - y_hat) ** 2)
        grads = tape.gradient(loss, siren.trainable_variables)
        optimizer.apply_gradients(zip(grads, siren.trainable_variables))
        
        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy().squeeze():0.4f}, step_size => {step_size:0.4f}"
            )

            bar.refresh()
    breakpoint()
    img_out = Image.fromarray(tf.reshape(y_hat*255, shape = [273, 365, 3]).numpy().astype(np.uint8))
    img_out.save("out_rgb.png")
    breakpoint()