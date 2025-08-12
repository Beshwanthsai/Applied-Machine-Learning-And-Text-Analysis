import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam


num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)


def run_rnn_experiment(seq_length):
    print(f"\nTraining RNN with sequence length: {seq_length}")
    x_train_pad = pad_sequences(x_train, maxlen=seq_length)
    x_test_pad = pad_sequences(x_test, maxlen=seq_length)

    model = Sequential([
        Embedding(input_dim=num_words, output_dim=32),
        SimpleRNN(32),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_pad, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=0)

    loss, accuracy = model.evaluate(x_test_pad, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


sequence_lengths = [50, 100, 200, 300, 500]
accuracies = [run_rnn_experiment(seq_len) for seq_len in sequence_lengths]

plt.figure(figsize=(8, 5))
plt.plot(sequence_lengths, accuracies, marker='o', linestyle='-', color='blue')
plt.title("Impact of Sequence Length on RNN Accuracy (IMDB)")
plt.xlabel("Sequence Length")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.show()