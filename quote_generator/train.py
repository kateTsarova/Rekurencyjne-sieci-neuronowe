from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Bidirectional

from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np

text = open("./database.txt", "r").read()

unused = ["#", "$", "%", "(", ")", "=", ";", ":", "*", "+", "£", "—", "’"]
for c in unused:
    text = text.replace(c, "")

vocab = sorted(set(text))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

quotes = text.split("\n")
quotes = [s + "\n" for s in quotes]

seq_length = 15
step = 6
sentences = []
next_chars = []

for quote in quotes:
    for i in range(0, len(quote) - seq_length, step):
        sentences.append(quote[i: i + seq_length])
        next_chars.append(quote[i + seq_length])
    sentences.append(quote[-seq_length:])
    next_chars.append(quote[-1])

x = np.zeros((len(sentences), seq_length, len(vocab)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char2idx[char]] = 1
    y[i, char2idx[next_chars[i]]] = 1

print('Build model...')
input_sequences = Input((seq_length, len(vocab)), name="input_sequences")
lstm = Bidirectional(LSTM(256, return_sequences=True, input_shape=(seq_length, len(vocab))), name='bidirectional')(
    input_sequences)
lstm = Dropout(0.1, name='dropout_bidirectional_lstm')(lstm)
lstm = LSTM(64, input_shape=(seq_length, len(vocab)), name='lstm')(lstm)
lstm = Dropout(0.1, name='drop_out_lstm')(lstm)

dense = Dense(15 * len(vocab), name='first_dense')(lstm)
dense = Dropout(0.1, name='drop_out_first_dense')(dense)
dense = Dense(5 * len(vocab), name='second_dense')(dense)
dense = Dropout(0.1, name='drop_out_second_dense')(dense)
dense = Dense(len(vocab), name='last_dense')(dense)

next_char = Activation('softmax', name='activation')(dense)

model = Model([input_sequences], next_char)
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit([x], y, batch_size=128, epochs=15)

model.fit([x], y, batch_size=2048, epochs=2)

model.fit([x], y, batch_size=1024, epochs=2)

model.save('./models/my_model.h5')
