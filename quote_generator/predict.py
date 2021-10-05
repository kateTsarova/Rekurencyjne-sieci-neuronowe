from nltk import word_tokenize
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input, Activation, Dense, Dropout
from tensorflow.keras.layers import LSTM, Bidirectional

from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed, GRU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import get_file
import numpy as np
import sys

seq_length = 15

text = open("./database.txt", "r").read()

unused = ["#", "$", "%", "(", ")", "=", ";", ":", "*", "+", "£", "—", "’"]
for c in unused:
    text = text.replace(c, "")

vocab = sorted(set(text))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

model = keras.models.load_model('models/my_model.h5')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_quote(sentence=None, diversity=0.8):
    if len(sentence) > seq_length:
        sentence = sentence[-seq_length:]
    elif len(sentence) < seq_length:
        sentence = ' ' * (seq_length - len(sentence)) + sentence

    generated = ''
    generated += sentence
    sys.stdout.write(generated)

    next_char = 'Empty'
    total_word = 0

    while ((next_char not in ['\n', '.']) & (total_word <= 500)):

        x_pred = np.zeros((1, seq_length, len(vocab)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char2idx[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = idx2char[next_index]

        if next_char == ' ':
            total_word += 1
        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()


generate_quote("You")
