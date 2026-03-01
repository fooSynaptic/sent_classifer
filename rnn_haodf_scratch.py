# -*- coding:utf-8 -*-
"""
Character-level RNN for Chinese Text Generation

This module implements a character-level RNN using MXNet for Chinese text generation.
The model is trained on medical question data and can generate new questions based on
provided prefixes.

Example:
    >>> python rnn_haodf_scratch.py
    
Note:
    Update the DATA_PATH, TAGS_PATH, and OUTPUT_PATH variables before running.
"""

import json
import sys
import os
import re
import random
from math import exp
import mxnet as mx
from mxnet import nd, gluon, autograd

# Configuration - Update these paths for your environment
DATA_PATH = 'C:/Users/QTC I7-1060/Desktop/ml_hu/data/result_clean.txt'
DATA_PATH_2 = 'C:/Users/QTC I7-1060/Desktop/ml_hu/data/lchen_result_clean.txt'
TAGS_PATH = 'C:/Users/QTC I7-1060/Desktop/ml_hu/data/tags_conditions_lung.json'
OUTPUT_PATH = 'C:/Users/QTC I7-1060/Desktop/ml_hu/data/que_generate.txt'

# Model hyperparameters
HIDDEN_DIM = 500
BATCH_SIZE = 32
NUM_STEPS = 10
LEARNING_RATE = 0.2
CLIPPING_THETA = 0.01
EPOCHS = 200
PRED_PERIOD = 20
PRED_LEN = 20

# Context (GPU if available, else CPU)
CTX = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()


def load_data(file_path: str, encoding: str = 'utf-8') -> list:
    """Load data from a text file."""
    with open(file_path, 'r', encoding=encoding) as f:
        return f.readlines()


def clean_questions(que_set: list) -> list:
    """Clean question text by removing unwanted patterns."""
    cleaned = []
    for question in que_set:
        q_clean = question.strip('好大夫在线网上咨询').strip('_').strip('\n')
        if '...' not in q_clean:
            cleaned.append(q_clean)
    return cleaned


def filter_by_length(data: list, min_len: int = 15) -> list:
    """Filter data by minimum length."""
    return [item for item in data if len(item) > min_len]


def preprocess_text(texts: list) -> list:
    """Preprocess text by removing special characters."""
    chars_to_remove = [' ', '-', '？', '?', '\n', '_', '.', ',']
    processed = []
    for text in texts:
        for char in chars_to_remove:
            text = text.replace(char, '')
        processed.append(text.strip())
    return processed


def build_vocab(corpus: str) -> tuple:
    """Build vocabulary mappings from corpus."""
    idx_to_char = list(set(corpus))
    char_to_idx = {char: idx for idx, char in enumerate(idx_to_char)}
    vocab_size = len(char_to_idx)
    return idx_to_char, char_to_idx, vocab_size


def data_iter_random(corpus_indices: list, batch_size: int, num_steps: int, ctx=None):
    """Random sampling data iterator."""
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        data = nd.array([_data(j * num_steps) for j in batch_indices], ctx=ctx)
        label = nd.array([_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield data, label


def data_iter_consecutive(corpus_indices: list, batch_size: int, num_steps: int, ctx=None):
    """Consecutive sampling data iterator."""
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size

    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i: i + num_steps]
        label = indices[:, i + 1: i + num_steps + 1]
        yield data, label


def get_inputs(data, vocab_size: int):
    """Convert data to one-hot encoding."""
    return [nd.one_hot(X, vocab_size) for X in data.T]


def get_params(vocab_size: int, hidden_dim: int, ctx):
    """Initialize model parameters."""
    std = 0.01
    
    W_xh = nd.random_normal(scale=std, shape=(vocab_size, hidden_dim), ctx=ctx)
    W_hh = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_h = nd.zeros(hidden_dim, ctx=ctx)
    W_hy = nd.random_normal(scale=std, shape=(hidden_dim, vocab_size), ctx=ctx)
    b_y = nd.zeros(vocab_size, ctx=ctx)

    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params


def rnn(inputs, state, *params):
    """RNN forward pass."""
    H = state
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return outputs, H


def predict_rnn(rnn_func, prefix: str, num_chars: int, params, hidden_dim: int, 
                ctx, idx_to_char: list, char_to_idx: dict, get_inputs_func, 
                is_lstm: bool = False):
    """Predict/generate text using the trained RNN."""
    state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    if is_lstm:
        state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    
    output = [char_to_idx[prefix[0]]]
    
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        
        if is_lstm:
            Y, state_h, state_c = rnn_func(
                get_inputs_func(X, len(char_to_idx)), state_h, state_c, *params
            )
        else:
            Y, state_h = rnn_func(
                get_inputs_func(X, len(char_to_idx)), state_h, *params
            )
        
        if i < len(prefix) - 1:
            next_input = char_to_idx[prefix[i + 1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    
    return ''.join([idx_to_char[i] for i in output])


def grad_clipping(params, theta: float, ctx):
    """Clip gradients to prevent explosion."""
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for p in params:
            norm += nd.sum(p.grad ** 2)
        norm = nd.sqrt(norm).asscalar()
        
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm
        else:
            for p in params:
                p.grad[:] *= 1


def train_and_predict_rnn(rnn_func, is_random_iter: bool, epochs: int, 
                          num_steps: int, hidden_dim: int, learning_rate: float,
                          clipping_theta: float, batch_size: int, pred_period: int,
                          pred_len: int, prefixes: list, get_params_func, 
                          get_inputs_func, ctx, corpus_indices: list, 
                          idx_to_char: list, char_to_idx: dict, is_lstm: bool = False):
    """Train and predict using RNN."""
    data_iter = data_iter_random if is_random_iter else data_iter_consecutive
    params = get_params_func(len(char_to_idx), hidden_dim, ctx)
    
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    ans_collect = []

    for epoch in range(1, epochs + 1):
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            if is_lstm:
                state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        
        train_loss, num_examples = 0, 0
        
        for data, label in data_iter(corpus_indices, batch_size, num_steps, ctx):
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                if is_lstm:
                    state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            
            with autograd.record():
                if is_lstm:
                    outputs, state_h, state_c = rnn_func(
                        get_inputs_func(data, len(char_to_idx)), state_h, state_c, *params
                    )
                else:
                    outputs, state_h = rnn_func(
                        get_inputs_func(data, len(char_to_idx)), state_h, *params
                    )
                
                label = label.T.reshape((-1,))
                outputs = nd.concat(*outputs, dim=0)
                loss = softmax_cross_entropy(outputs, label)
            
            loss.backward()
            grad_clipping(params, clipping_theta, ctx)
            
            for p in params:
                p[:] -= learning_rate * p.grad / batch_size
            
            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size

        if epoch % pred_period == 0:
            try:
                perplexity = exp(train_loss / num_examples)
                print(f"Epoch {epoch}. Perplexity {perplexity:.4f}")
            except OverflowError:
                print(f"Overflow! Loss: {train_loss}; Examples: {num_examples}")
                perplexity = float('inf')
            
            for seq in prefixes:
                try:
                    result = predict_rnn(
                        rnn_func, seq, pred_len, params, hidden_dim, ctx,
                        idx_to_char, char_to_idx, get_inputs_func, is_lstm
                    )
                    print(f'- {seq} -> {result}')
                    ans_collect.append(result.split()[0] if ' ' in result else result[:20])
                    
                    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                        f.write(result + '\n')
                except Exception as e:
                    print(f'Theme {seq} failed: {e}')
            print()
    
    return ans_collect


def load_themes(tags_path: str) -> list:
    """Load theme tags from JSON file."""
    with open(tags_path, encoding='utf-8') as f:
        setting = json.load(f)
    
    themes = []
    for item in setting:
        if item.get('tags'):
            themes.extend(item['tags'].split(','))
    
    return themes


def main():
    """Main function to run the RNN training and prediction."""
    print(f'Using context: {CTX}')
    
    wenda = load_data(DATA_PATH)
    wenda.extend(load_data(DATA_PATH_2))
    print(f'Total samples: {len(wenda)}')
    
    ans_fined = clean_questions(wenda)
    fined_ques = filter_by_length(ans_fined, min_len=15)
    answer = preprocess_text(fined_ques)
    
    print(f'After cleaning: {len(answer)} questions')
    
    corpus_chars = ' '.join(answer)
    idx_to_char, char_to_idx, vocab_size = build_vocab(corpus_chars)
    
    print(f'Vocabulary size: {vocab_size}')
    
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    
    theme_collect = load_themes(TAGS_PATH)
    
    valid_themes = []
    for theme in theme_collect:
        if all(char in char_to_idx for char in theme):
            valid_themes.append(theme)
    
    random.shuffle(valid_themes)
    prefixes = valid_themes[:99]
    
    print(f'Using {len(prefixes)} prefixes for prediction')
    
    result = train_and_predict_rnn(
        rnn_func=rnn,
        is_random_iter=False,
        epochs=EPOCHS,
        num_steps=NUM_STEPS,
        hidden_dim=HIDDEN_DIM,
        learning_rate=LEARNING_RATE,
        clipping_theta=CLIPPING_THETA,
        batch_size=BATCH_SIZE,
        pred_period=PRED_PERIOD,
        pred_len=PRED_LEN,
        prefixes=prefixes,
        get_params_func=lambda vs, hd, c: get_params(vs, hd, c),
        get_inputs_func=lambda d: get_inputs(d, len(char_to_idx)),
        ctx=CTX,
        corpus_indices=corpus_indices,
        idx_to_char=idx_to_char,
        char_to_idx=char_to_idx
    )
    
    print('Training completed!')
    print(f'Results saved to: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
