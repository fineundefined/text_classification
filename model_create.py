import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional,Dense,Dropout,LSTM



def hate_class(vocab_size, max_length):
    model = Sequential([
        Embedding(vocab_size, 64,input_length=max_length),
        Bidirectional(LSTM(64)),
        Dense(32,activation="relu"),
        Dropout(0.5),
        Dense(3, activation="softmax")
    ])
    return model
