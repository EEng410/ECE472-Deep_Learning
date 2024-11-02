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
        self.d_model = d_model
    def __call__(self, input_seq):
        k = tf.cast(tf.range(0, len(input_seq), 1), dtype = tf.float32)
        i = tf.range(0, self.d_model/2, 1)/self.d_model
        encoding = []
        K, I = tf.meshgrid(k, i)
        
        encoding_sin = tf.sin(tf.divide(K, 10000**I))
        encoding_cos = tf.cos(tf.divide(K, 10000**I))
        # breakpoint()
        encoding = tf.transpose(tf.reshape(tf.stack([encoding_sin, encoding_cos], axis = 1), [len(input_seq)
, -1]))
        # encoding returns as seq_len, d_model
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


# class Transformer(tf.Module):
#     def __init__(self, d_model, n_heads, n_layers, vocab_size):
#         self.d_model = d_model
#         self.n_heads = n_heads
#         # Embedding layer will project to batch_size, seq_len, d_model
#         self.residual_layers = self.
#     def __call__(self, input):



if __name__ == '__main__':

    corpus = "I love to eat, eat, eat. Apples and bananas!"
    tokenizer = Tokenizer(corpus)
    d_model = 20
    positional_encoding = PositionalEncoder(d_model)
    # breakpoint()
    tokenized = tokenizer.tokenize("I love, to !eat")
    # perform a one_hot encoding and 
    batch_size = 1
    tokenized_one_hot = tf.one_hot(tokenized, tokenizer.vocab_size)
    
    encoding_test = positional_encoding(tokenized)
    
    tokenized_one_hot = repeat(tokenized_one_hot, "seq_len vocab_size -> batch_size seq_len vocab_size", batch_size = batch_size)
    # need to one hot and add a batch dim to input
    multi_head_attention = ResidualBlock(5, d_model, d_model)
    embedding_layer = Linear(tokenizer.vocab_size, d_model)
    multi_head_attention(embedding_layer(tokenized_one_hot))
    # self_attention = SelfAttention()
