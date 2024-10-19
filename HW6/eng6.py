import tensorflow as tf
from einops import einsum
import re

class SelfAttention(tf.Module):
    def __init__(self, vocab_size, num_outputs, d_model, d_values):
        rng = tf.random.get_global_generator()
        self.W_Q = tf.Variable(
            rng.normal(vocab_size, d_model),
            trainable = True,
            name = "Self/Attention/Query_Weights"
        )
        self.W_K = tf.Variable(
            rng.normal(vocab_size, d_model),
            trainable = True,
            name = "Self/Attention/Keys_Weights"
        )
        self.W_V = tf.Variable(
            rng.normal(vocab_size, d_values),
            trainable = True,
            name = "Self/Attention/Values_Weights"
        )
        self.d_model = d_model
        self.d_values = d_values

    def __call__(self, x):
        # Input should be [batch_size, sequence_length, vocab_size]
        Q = einsum(x, self.W_Q, 'batch_size seq_len vocab_size, vocab_size d_model -> batch_size seq_len d_model')
        K = einsum(x, self.W_K, 'batch_size seq_len vocab_size, vocab_size d_model -> batch_size seq_len d_model')
        V = einsum(x, self.W_V, 'batch_size seq_len vocab_size, vocab_size d_model -> batch_size seq_len d_model')

        attention_mat = tf.nn.softmax(einsum(Q, K, 'batch_size seq_len d_model, batch seq_len d_model -> batch_size vocab_size vocab_size') / (self.d_model**0.5), axis = 1)   
        out = einsum(attention_mat, V, 'batch_size vocab_size vocab_size, vocab_size d_values')     
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
    
if __name__ == '__main__':
    from datasets import load_dataset
    corpus = "I love to eat, eat, eat. Apples and bananas!"
    tokenizer = Tokenizer(corpus)
    # breakpoint()
    tokenized = tokenizer.tokenize("I love, to !eat")
    print(tokenizer.detokenize(tokenized))
    breakpoint()
    # self_attention = SelfAttention()
