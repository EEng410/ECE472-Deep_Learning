import tensorflow as tf
from einops import einsum
from einops import rearrange
from einops import repeat
import re

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
        # Input should be of dimension batch_size, seq_len, input_dim
        z = einsum(x, self.w, 'batch_size seq_len num_inputs, num_inputs num_outputs -> batch_size seq_len num_outputs')
        if self.bias:
            z = einsum(z, self.w, 'batch_size seq_len num_outputs, one num_outputs -> batch_size seq_len num_outputs')
        return z

class SelfAttention(tf.Module):
    def __init__(self, d_in, d_model, d_values, max_seq_len = 10):
        rng = tf.random.get_global_generator()
        stddev = tf.math.sqrt(2 / (d_in + d_model))
        stddev_values = tf.math.sqrt(2 / (d_in + d_values))
        self.W_Q = tf.Variable(
            rng.normal(shape=[d_in, d_model], stddev=stddev),
            trainable = True,
            name = "Self/Attention/Query_Weights"
        )
        self.W_K = tf.Variable(
            rng.normal(shape=[d_in, d_model], stddev=stddev),
            trainable = True,
            name = "Self/Attention/Keys_Weights"
        )
        self.W_V = tf.Variable(
            rng.normal(shape=[d_in, d_model], stddev=stddev_values),
            trainable = True,
            name = "Self/Attention/Values_Weights"
        )
        self.d_model = d_model
        self.d_values = d_values

    def __call__(self, x):
        x = tf.cast(x, dtype = tf.float32)
        # Input should be [batch_size, sequence_length, vocab_size]
        # breakpoint()
        mask = tf.cast(self.get_causal_attention_mask(x), dtype = tf.float32)

        Q = einsum(x, self.W_Q, 'batch_size seq_len d_in, d_in d_model -> batch_size seq_len d_model')
        K = einsum(x, self.W_K, 'batch_size seq_len d_in, d_in d_model -> batch_size seq_len d_model')
        V = einsum(x, self.W_V, 'batch_size seq_len d_in, d_in d_model -> batch_size seq_len d_model')
        # Need to implement masking here:
        attention_mat = einsum(Q, K, 'batch_size seq_len d_model, batch seq_len_2 d_model -> batch_size seq_len seq_len_2') / (self.d_model**0.5)
        attention_mat = tf.nn.softmax(attention_mat, axis = 1)   * mask

        # Output will be batch_size, seq_len, d_values

        out = einsum(attention_mat, V, 'batch_size seq_len seq_len, batch_size seq_len d_values -> batch_size seq_len d_values')     
        return out
    # This is sourced from https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, None]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.convert_to_tensor([1, 1])],
            axis=0,
        )
        return tf.tile(mask, mult)
    
class Tokenizer(tf.Module):
    def __init__(self, corpus):
        #building a vocabulary
        self.invalid_syntax = [',', '.', '!', '?']
        words = corpus
        for punc in self.invalid_syntax:
            # breakpoint()
            words = words.replace(punc, '')
        words = words.lower().split()
        #send all the valid words to lower_case
        self.vocab = list(set(words))
        self.vocab_size = len(self.vocab)
        # Create word-to-index and index-to-word mappings
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}

        # Tensor of indices for vocabulary
        self.index_list = tf.constant(range(self.vocab_size))
    def tokenize(self, input):
        # Clean the input (remove punctuation and make it lowercase)
        words = input
        for naughty_boys in self.invalid_syntax:
            words = words.replace(naughty_boys, '')
        words = words.lower().split()
        # breakpoint()
        # Map each word to its corresponding index
        tokens = [self.word2idx[word] for word in words if word in self.vocab]

        return tf.constant(tokens, dtype=tf.int32)

    def detokenize(self, tokens):
        # Map each index back to its corresponding word without puncuation
        words = [self.idx2word[token.numpy()] for token in tokens]
        return ' '.join(words)

class MultiHeadAttention(tf.Module):
    def __init__(self, n_heads, d_model, d_values):
        self.heads = []
        # Let's just assume input/output dimensions are the same...
        self.n_heads = n_heads
        for i in range(n_heads):
            self.heads.append(SelfAttention(int(d_model/n_heads), int(d_model/n_heads), int(d_values/n_heads)))
        self.d_model = d_model
    def __call__(self, input):
        #assuming input is tokenized
        head_outputs = []
        input_head_divided = rearrange(input, 'batch_size seq_len (head d_model_by_h) -> batch_size seq_len head d_model_by_h', head = self.n_heads, d_model_by_h = self.d_model // self.n_heads)
        for i in  range(self.n_heads):
            head_outputs.append(self.heads[i](input_head_divided[:, :, i, :]))
            
        # head_outputs come out as n_heads, batch_size, seq_len, d_model / h
        out = tf.concat(head_outputs, 2)
        # out comes out as batch_size, seq_len, d_model
        return out

class PositionalEncoder(tf.Module):
    def __init__(self, d_model):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model

    def __call__(self, input_seq):
        batch_size, seq_len = tf.shape(input_seq)[0], tf.shape(input_seq)[1]
        k = tf.cast(tf.range(seq_len), dtype=tf.float32)
        i = tf.range(0, self.d_model // 2, dtype=tf.float32) / self.d_model
        K, I = tf.meshgrid(k, i)
        K = tf.transpose(K)
        I = tf.transpose(I)
        encoding_sin = tf.sin(tf.divide(K, 10000**I))
        encoding_cos = tf.cos(tf.divide(K, 10000**I))

        # Stack sin and cos encoding and reshape to (seq_len, d_model)
        encoding = tf.reshape(tf.stack([encoding_sin, encoding_cos], axis=2), [seq_len, self.d_model])
        # Expand the encoding to (batch_size, seq_len, d_model) by repeating along the batch dimension
        encoding = tf.tile(tf.expand_dims(encoding, 0), [batch_size, 1, 1])

        return encoding


class LayerNorm(tf.Module):
    def __init__(self, input_shape, epsilon):
        self.gamma = tf.Variable(
            tf.ones(shape = input_shape, dtype = tf.float32), 
            trainable = True,
            name = "LN/gamma"
        )
        self.beta = tf.Variable(
            tf.zeros(shape = input_shape, dtype = tf.float32), 
            trainable = True,
            name = "LN/beta"
        )
        self.epsilon = epsilon
    def __call__(self, input):
        # Input should be of the shape batch_size, seq_len, d_model
        mean = tf.math.reduce_mean(input, axis = 2, keepdims = True)
        var  = tf.math.reduce_variance(input, axis = 2, keepdims = True)
        # var = repeat(var, 'batch_size seq_len -> batch_size seq_len d_model', d_model = self.gamma.shape[1])
        input_normalized = (input-mean) / (var**0.5 + self.epsilon)

        # out = einsum(input_normalized, self.gamma,  'batch_size seq_len d_model, batch_size d_model -> batch_size seq_len d_model')
        out = self.gamma * input_normalized + self.beta

        return out

class ResidualBlock(tf.Module):
    def __init__(self, n_heads, d_model, d_values):
        self.multi_head_attention = MultiHeadAttention(n_heads, d_model, d_values)
        batch_size = 1
        self.layer_norm = LayerNorm([batch_size, d_model], 1e-5)

        self.output_layer = Linear(d_model, d_model)
        
    def __call__(self, input):
        input_attended = self.multi_head_attention(input)
        input_normalized = self.layer_norm(input_attended)
        # Implement the first residual connection around the multiheaded attention
        z = input + input_normalized

        z_proj = self.output_layer(z)
        out = z_proj + z
        return out


class Transformer(tf.Module):
    def __init__(self, d_model, n_heads, n_layers, vocab_size):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        # Embedding layer will project to batch_size, seq_len, d_model
        self.residual_layers = []
        self.embedding_layer = Linear(vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model)
        for i in range(n_layers):
            self.residual_layers.append(ResidualBlock(n_heads, d_model, d_model))
        self.output_layer = Linear(d_model, vocab_size)

        # Output layer will need to be softmaxxed
    def __call__(self, input):
        # Assume input has been tokenized and has yet to be one-hot encoded
        position_encoding = self.positional_encoder(input)

        input_one_hot = tf.one_hot(input, self.vocab_size)
        input_embedded = self.embedding_layer(input_one_hot)
        # Embed input and apply positional encoding
        input_embedded = input_embedded + position_encoding
        x = input_embedded
        for i in range(self.n_layers):
            x = self.residual_layers[i](x)
        z = self.output_layer(x)
        # axis = 2 must be softmaxxed
        z = tf.nn.softmax(z, axis = 2)

        return z

def transformer_make_prediction(transformer, tokenizer, input_seq):
    seq = tokenizer.tokenize(input_seq)
    seq_batch = repeat(seq, "seq_len -> batch_size seq_len", batch_size = 1)
    out = transformer(seq_batch)
    out_tokens = tf.argmax(out, axis = 2)[0]

    return tokenizer.detokenize(out_tokens)

if __name__ == '__main__':
    import argparse

    import matplotlib.pyplot as plt
    from tqdm import trange
    corpus = "Free me from this flesh prison. I yearn for freedom from the confines of skin and bone."
    tokenizer = Tokenizer(corpus)
    d_model = 20
    n_heads = 5
    n_layers = 2
    positional_encoding = PositionalEncoder(d_model)
    # breakpoint()
    tokenized = tokenizer.tokenize("Free me from this flesh prison. I yearn for freedom from the confines of skin and")
    truth = tokenizer.tokenize("me from this flesh prison. I yearn for freedom from the confines of skin and bone.")
    # perform a one_hot encoding and 
    batch_size = 1
    tokenized_one_hot = tf.one_hot(tokenized, tokenizer.vocab_size)
    cce = tf.keras.losses.CategoricalCrossentropy()
    # encoding_test = positional_encoding(tokenized)
    
    tokenized = repeat(tokenized, "seq_len -> batch_size seq_len", batch_size = batch_size)
    truth = repeat(truth, "seq_len -> batch_size seq_len", batch_size = batch_size)
    transformer = Transformer(d_model, n_heads, n_layers, tokenizer.vocab_size)


    test = transformer(tokenized)
    truth = tf.one_hot(truth, tokenizer.vocab_size)

    num_iters = 1000
    step_size = 0.005
    decay_rate = 0.999
    gamma = 0.0004
    refresh_rate = 10

    optimizer = tf.keras.optimizers.AdamW(learning_rate = step_size, weight_decay = gamma)
    bar = trange(num_iters)

    for i in bar:
        x_batch = tokenized
        y_batch = truth
        with tf.GradientTape() as tape:

            y_hat = transformer(x_batch)
            loss =  cce(y_batch, y_hat)
            loss = tf.math.reduce_mean(loss)

        grads = tape.gradient(loss, transformer.trainable_variables)

        optimizer.apply_gradients(zip(grads, transformer.trainable_variables))

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy().squeeze():0.4f}, step_size => {step_size:0.4f}"
            )

            bar.refresh()
    breakpoint()
    # need to one hot and add a batch dim to input
    # multi_head_attention = ResidualBlock(5, d_model, d_model)
    # embedding_layer = Linear(tokenizer.vocab_size, d_model)
    # multi_head_attention(embedding_layer(tokenized_one_hot))
    # self_attention = SelfAttention()
