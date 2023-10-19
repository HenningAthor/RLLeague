"""
Script to learn an autoencoder.
"""
from autoencoder.learn import learn_rl

if __name__ == '__main__':
    learn_rl(path='recorded_data/out/', epochs=50, hidden_size=2, output_path='autoencoder/autoencoder_2')
    learn_rl(path='recorded_data/out/', epochs=50, hidden_size=4, output_path='autoencoder/autoencoder_4')
    learn_rl(path='recorded_data/out/', epochs=50, hidden_size=8, output_path='autoencoder/autoencoder_8')
    learn_rl(path='recorded_data/out/', epochs=50, hidden_size=16, output_path='autoencoder/autoencoder_16')
    learn_rl(path='recorded_data/out/', epochs=50, hidden_size=32, output_path='autoencoder/autoencoder_32')
