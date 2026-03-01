# sent_classifier - Sentence Classification with RNN

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![MXNet](https://img.shields.io/badge/MXNet-1.5+-green.svg)](https://mxnet.apache.org/)

A character-level RNN implementation for sentence generation and classification using Apache MXNet (Gluon API).

## <img src=".github/icons/book.svg" width="16" height="16" alt="book"> Overview

This project implements a character-level Recurrent Neural Network (RNN) for Chinese text generation. The model learns patterns from a corpus of medical questions and can generate new questions based on given prefixes.

## <img src=".github/icons/rocket.svg" width="16" height="16" alt="rocket"> Quick Start

### Prerequisites

```bash
pip install mxnet numpy jieba
```

### Data Preparation

The model expects data files containing Chinese text (questions/answers) in the following format:
- One entry per line
- UTF-8 encoding

Update the file paths in `rnn_haodf_scratch.py` to point to your data files.

### Training

```bash
python rnn_haodf_scratch.py
```

## <img src=".github/icons/folder.svg" width="16" height="16" alt="folder"> Project Structure

```
sent_classifer/
├── rnn_haodf_scratch.py    # Main RNN implementation
├── README.md                # This file
└── a.txt                    # Sample data
```

## 🧠 Model Architecture

```
Input (Character Index)
    ↓
One-Hot Encoding
    ↓
RNN Cell (tanh activation)
    ↓
Output Layer
    ↓
Softmax (Vocabulary Distribution)
```

### Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `hidden_dim` | 500 | Hidden state dimension |
| `batch_size` | 32 | Training batch size |
| `num_steps` | 10 | Sequence length |
| `learning_rate` | 0.2 | Initial learning rate |
| `clipping_theta` | 0.01 | Gradient clipping threshold |
| `epochs` | 200 | Number of training epochs |

## <img src=".github/icons/chart.svg" width="16" height="16" alt="chart"> Training Details

### Data Iterator

Two sampling methods are implemented:

1. **Random Sampling** (`data_iter_random`): Shuffles examples each epoch
2. **Consecutive Sampling** (`data_iter_consecutive`): Sequential batches with state persistence

### Gradient Clipping

Prevents gradient explosion by clipping gradients with norm > theta:

```python
grad_clipping(params, theta, ctx)
```

### Loss Function

Softmax Cross-Entropy Loss with perplexity evaluation.

## <img src=".github/icons/note.svg" width="16" height="16" alt="note"> Usage Example

```python
import mxnet as mx
from mxnet import nd, gluon

# Load and preprocess data
# (Update file paths in the script)

# Train the model
train_and_predict_rnn(
    rnn=rnn,
    is_random_iter=False,
    epochs=200,
    num_steps=10,
    hidden_dim=500,
    learning_rate=0.2,
    clipping_theta=0.01,
    batch_size=32,
    pred_period=20,
    pred_len=20,
    seqs=theme_list,
    get_params=get_params,
    get_inputs=get_inputs,
    ctx=mx.gpu(),
    corpus_indices=corpus_indices,
    idx_to_char=idx_to_char,
    char_to_idx=char_to_idx
)

# Generate predictions
result = predict_rnn(
    rnn=rnn,
    prefix="症状",
    num_chars=50,
    params=params,
    hidden_dim=500,
    ctx=mx.gpu(),
    idx_to_char=idx_to_char,
    char_to_idx=char_to_idx,
    get_inputs=get_inputs
)
```

## <img src=".github/icons/wrench.svg" width="16" height="16" alt="wrench"> Customization

### Update Data Paths

Modify these variables in `rnn_haodf_scratch.py`:

```python
DATA_PATH = 'path/to/your/data.txt'
TAGS_PATH = 'path/to/your/tags.json'
OUTPUT_PATH = 'path/to/output.txt'
```

### Adjust Model Size

Change `hidden_dim` in the hyperparameters section:

```python
HIDDEN_DIM = 500  # Increase for more capacity, decrease for faster training
```

## <img src=".github/icons/warning.svg" width="16" height="16" alt="warning"> Known Issues

- Hard-coded file paths need to be updated for your environment
- Uses deprecated MXNet APIs (asscalar, nd API)
- No command-line argument support

## <img src=".github/icons/book.svg" width="16" height="16" alt="book"> References

- [MXNet Gluon Documentation](https://mxnet.apache.org/api/python/docs/tutorials/)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
