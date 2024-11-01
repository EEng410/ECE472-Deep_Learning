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
            z = einsum(x, self.w, 'batch_size seq_len num_outputs, one num_outputs -> batch_size seq_len num_outputs')
        return z

class SelfAttention(tf.Module):
    def __init__(self, d_in, d_model, d_values):
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
        Q = einsum(x, self.W_Q, 'batch_size seq_len d_in, d_in d_model -> batch_size seq_len d_model')
        K = einsum(x, self.W_K, 'batch_size seq_len d_in, d_in d_model -> batch_size seq_len d_model')
        V = einsum(x, self.W_V, 'batch_size seq_len d_in, d_in d_model -> batch_size seq_len d_model')

        attention_mat = tf.nn.softmax(einsum(Q, K, 'batch_size seq_len d_model, batch seq_len_2 d_model -> batch_size seq_len seq_len_2') / (self.d_model**0.5), axis = 1)   
        # Output will be batch_size, seq_len, d_values
        out = einsum(attention_mat, V, 'batch_size seq_len seq_len, batch_size seq_len d_values -> batch_size seq_len d_values')     
        return out
    
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
            self.heads.append(SelfAttention(d_model, d_model, d_values))
        rng = tf.random.get_global_generator()
        stddev = tf.math.sqrt(2 / (d_model + d_model))
        
    def __call__(self, input):
        #assuming input is tokenized
        head_outputs = []
        input_head_divided = rearrange(input, 'batch_size, seq_len, d_model -> batch_size, seq_len, head, d_model_by_h', head = self.n_heads)
        for i in  range(self.n_heads):
            head_outputs.append(self.heads[i](tf.squeeze(input_head_divided[i], axis = 2)))
            #heads come out as batch_size, seq_len, d_model
        breakpoint()
        # head_outputs come out as n_heads, batch_size, seq_len, d_model / h
        out = tf.concat(head_outputs, 3)
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
    def __init__(self, input_shape, gamma, beta, epsilon):
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
class ResidualBlock(tf.Module):
    def __init__(self, d_model, n_heads):
        self.multi_head_attention = MultiHeadAttention(n_heads, d_model, d_model, d_model)
        self.layer_norm = LayerNorm()
    def __call__(self, input):

class Transformer(tf.Module):
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        # Embedding layer will project to batch_size, seq_len, d_model
        self.embedding_layer = Linear(1, d_model)
        self.positional_encoding = PositionalEncoder(d_model)
        self.multi_head_attention_layer = MultiHeadAttention(self.n_heads, d_model, d_model, d_model)
    def __call__(self, input):


if __name__ == '__main__':
    # from datasets import load_dataset
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
    multi_head_attention = MultiHeadAttention(5, tokenizer.vocab_size, d_model, 23)
    
    multi_head_attention(tokenized_one_hot)
    # self_attention = SelfAttention()
