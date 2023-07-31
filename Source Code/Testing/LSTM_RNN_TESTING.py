import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

model_filename = 'trained_model.h5'
loaded_model = load_model(model_filename)

tokenizer = Tokenizer(char_level=True)
label_encoder = LabelEncoder()

tokenizer_json = 'tokenizer_config.json'
label_encoder_file = 'label_encoder.npy'

with open(tokenizer_json, 'r') as f:
    tokenizer_config = json.load(f)
tokenizer.word_index = json.loads(tokenizer_config['word_index'])

label_encoder.classes_ = np.load(label_encoder_file, allow_pickle=True)

num_test_cases = int(input("Enter the number of ciphertexts you want to test : "))

for i in range(num_test_cases):
    user_input = input("Enter cipher text {} for testing: ".format(i + 1))
    user_input = user_input.lower()
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts([user_input])
    user_input_tokenized = tokenizer.texts_to_sequences([user_input])
    max_sequence_length = 40
    user_input_padded = pad_sequences(user_input_tokenized, maxlen=max_sequence_length)
    prediction_one_hot = loaded_model.predict(user_input_padded)
    predicted_label_numeric = np.argmax(prediction_one_hot, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_label_numeric)[0]
    print("Predicted Label:", predicted_label)
