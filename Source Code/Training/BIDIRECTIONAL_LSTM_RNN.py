import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def add_random_noise(texts, noise_level=0.1):
    noisy_texts = []
    for text in texts:
        noisy_text = ''.join([c if np.random.rand() > noise_level else chr(ord(c) + np.random.randint(-3, 3)) for c in text])
        noisy_texts.append(noisy_text)
    return noisy_texts

file_path = 'training_dataset.xlsx'
training_data = pd.read_excel(file_path)

cipher_texts_augmented = add_random_noise(training_data.iloc[:, 0].values)

cipher_texts_combined = np.concatenate((training_data.iloc[:, 0].values, cipher_texts_augmented))

labels_text_combined = np.concatenate((training_data.iloc[:, 1].values, training_data.iloc[:, 1].values))

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(cipher_texts_combined)
cipher_texts_tokenized = tokenizer.texts_to_sequences(cipher_texts_combined)

max_sequence_length = 40
cipher_texts_padded = pad_sequences(cipher_texts_tokenized, maxlen=max_sequence_length)

label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(labels_text_combined)

x_train, x_test, y_train, y_test = train_test_split(cipher_texts_padded, labels_numeric, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

def create_model(units_lstm=128, dropout_rate=0.5):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
    model.add(Bidirectional(LSTM(units=units_lstm, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units=units_lstm, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units=units_lstm)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

units_lstm_values = [64, 128, 256]
dropout_rate_values = [0.3, 0.5, 0.7]

best_accuracy = 0.0
best_model = None

for units_lstm in units_lstm_values:
    for dropout_rate in dropout_rate_values:
        print(f"Training model with units_lstm = {units_lstm} , dropout_rate = {dropout_rate}")

        model = create_model(units_lstm=units_lstm, dropout_rate=dropout_rate)

        history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val),
                            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)])

        train_accuracy = history.history['accuracy'][-1]
        test_accuracy = history.history['val_accuracy'][-1]

        print("Training Accuracy : {:.2f}%".format(train_accuracy * 100))
        print("Validation Accuracy : {:.2f}%".format(test_accuracy * 100))

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model

y_pred = best_model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
print("Test Set Classification Report : ")
print(classification_report(y_test, y_pred_labels))

conf_matrix = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix : ")
print(conf_matrix)

model_filename = 'trained_model.h5'
best_model.save(model_filename)
print("Trained model has been saved to : ", model_filename)

tokenizer_json = 'tokenizer_config.json'
tokenizer_config = tokenizer.get_config()
with open(tokenizer_json, 'w') as f:
    json.dump(tokenizer_config, f)

label_encoder_file = 'label_encoder.npy'
np.save(label_encoder_file, label_encoder.classes_)

