import tensorflow as tf
from sklearn.inspection import DecisionBoundaryDisplay

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

        return z

class MLP(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, hidden_activation = tf.math.sin, output_activation = tf.identity, bias=True):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.bias = bias
        self.input_layer = SineLayer(num_inputs, hidden_layer_width)

        self.hidden_layers = [SineLayer(hidden_layer_width, hidden_layer_width) for i in range(num_hidden_layers)]

        self.output_layer = SineLayer(hidden_layer_width, num_outputs)


    def __call__(self, x):
        z = tf.cast(x, 'float32')
        z = tf.transpose(z)

        z = self.input_layer(z)

        raise KeyError("finna bust")
       
        for layer in range(self.num_hidden_layers):
            z = self.hidden_activation(self.hidden_layers[layer](z))
        p = self.output_activation(self.output_layer(z))
        return p