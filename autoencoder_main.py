"""
Script to learn an autoencoder.
"""
from autoencoder.learn import learn_rl, learn_mnist


if __name__ == '__main__':
    learn_rl(epochs=40, hidden_size=8, output_path='autoencoder_8')

