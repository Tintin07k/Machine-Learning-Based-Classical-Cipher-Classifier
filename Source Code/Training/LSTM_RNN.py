import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import json

def read_cipher_text(file_path):
    df = pd.read_excel(file_path)
    return df['Cipher Text'], df['Cipher Label']

cipher_texts, labels_text = read_cipher_text('training_dataset.xlsx')

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(cipher_texts)
cipher_texts_tokenized = tokenizer.texts_to_sequences(cipher_texts)

max_sequence_length = 40
cipher_texts_padded = pad_sequences(cipher_texts_tokenized, maxlen=max_sequence_length, padding='post', truncating='post')

label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(labels_text)

x_train, x_test, y_train, y_test = train_test_split(cipher_texts_padded, labels_numeric, test_size=0.2, random_state=42)

rnn = Sequential()

rnn.add(LSTM(units=45, return_sequences=True, input_shape=(max_sequence_length, 1)))

rnn.add(Dropout(0.2))

for i in [True, True, False]:
    rnn.add(LSTM(units=45, return_sequences=i))
    rnn.add(Dropout(0.2))

num_unique_labels = len(label_encoder.classes_)
rnn.add(Dense(units=num_unique_labels, activation='softmax'))

rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = rnn.fit(x_train, y_train, epochs=2, batch_size=32, validation_data=(x_test, y_test))

train_accuracy = history.history['accuracy'][-1]
test_accuracy = history.history['val_accuracy'][-1]

print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))

model_filename = 'trained_model.h5'
rnn.save(model_filename)
print("Trained model has been saved to :", model_filename)

tokenizer_json = 'tokenizer_config.json'
tokenizer_config = tokenizer.get_config()
with open(tokenizer_json, 'w') as f:
    json.dump(tokenizer_config, f)

label_encoder_file = 'label_encoder.npy'
np.save(label_encoder_file, label_encoder.classes_)
