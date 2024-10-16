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

class MLP(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, hidden_activation = tf.identity, output_activation = tf.identity, bias=True, dropout_prob = 0.2):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_prob = dropout_prob
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

    def __call__(self, x, inference = False):
        z = tf.cast(x, 'float32')
        z = tf.transpose(z)
        z = self.input_layer(z)

        for layer in range(self.num_hidden_layers):
            z = self.hidden_activation(self.hidden_layers[layer](z))
            if inference == False:
                z = tf.nn.dropout(z, self.dropout_prob)

        p = self.output_activation(self.output_layer(z))

        # if p >= 0.5:
        #     return 1
        # if p < 0.5:
        #     return 0
        return p
    

class Classifier(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, hidden_activation = tf.nn.leaky_relu, output_activation = tf.nn.softmax, dropout_prob = 0.2, bias=True):
        self.mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, hidden_activation, output_activation, bias = bias)
        self.dropout_prob = dropout_prob
    def __call__(self, x, inference = False):
        #assume x is an input string. We first encode
        # Use MLP to arrive at ouptut probabilities
        x_encoded = tf.transpose(tf.Variable(x), [1, 0])
        z = self.mlp(x_encoded, inference)
        return z

def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        # breakpoint()
        var.assign_sub(step_size * grad)

def accuracy(truth, set, classifier):
    preds = tf.argmax(classifier(set, inference = True), axis = 1)
    truth = tf.argmax(truth, axis = 1)
    accuracy = tf.math.count_nonzero(tf.equal(truth,preds)) / preds.shape[0] * 100
    return accuracy

if __name__ == "__main__":
    import argparse
    import numpy as np
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from tqdm import trange
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    num_samples = 120000
    num_inputs = 384 # size of embeddings. Consistent because of padding
    num_outputs = 4
    num_iters = 1000
    step_size = 0.005
    decay_rate = 0.999
    batch_size = 128
    num_hidden_layers = 5
    hidden_layer_width = 256

    refresh_rate = 10
    classifier = Classifier(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)
    bar = trange(num_iters)
    gamma = 0.0004
    optimizer = tf.keras.optimizers.AdamW(learning_rate = step_size, weight_decay = gamma)
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_vec = []
    accuracy_test = 0.00
    ds = load_dataset("fancyzhx/ag_news")
    transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    test_labels = tf.one_hot(ds['test']['label'], 4)
    # test_embs = transformer.encode(ds['test']['text'])
    for i in bar:
        breakpoint()
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        ##Embeddings
        batch = ds['train'].select(batch_indices)
        x_batch = transformer.encode(batch['text'])
        y_batch = tf.one_hot(batch['label'], 4)
        y_batch = tf.cast(y_batch, 'float32')
        with tf.GradientTape() as tape:
            
            y_hat = classifier(x_batch)
            loss =  cce(y_batch, y_hat)
            loss = tf.math.reduce_mean(loss)

        grads = tape.gradient(loss, classifier.trainable_variables)

        optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
        # if i % 50 == 0:
            # accuracy_test = accuracy(test_labels, test_embs, classifier)

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy().squeeze():0.4f}, Accuracy => {accuracy_test:0.4f}, step_size => {step_size:0.4f}"
            )

            bar.refresh()
