from .base import NeuralNetwork
from .dense import MLP, CNN, RNN
from .sparse import SparseMLP, SparseCNN, SparseRNN

def get_neural_network(name: str) -> NeuralNetwork:
    name = name.lower()
    if name == 'mlp':
        return MLP
    elif name == 'cnn':
        return CNN
    elif name == 'rnn':
        return RNN
    elif name == 'sparse_mlp':
        return SparseMLP
    elif name == 'sparse_cnn':
        return SparseCNN
    elif name == 'sparse_rnn':
        return SparseRNN
    else:
        raise ValueError('Unknown model name: {0}'.format(name))
