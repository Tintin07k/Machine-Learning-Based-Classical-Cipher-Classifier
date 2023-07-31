from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model_filename = 'trained_model.h5'
loaded_model = load_model(model_filename)

tokenizer_filename = 'tokenizer_config.json'
with open(tokenizer_filename, 'r') as f:
    tokenizer_json = f.read()

from tensorflow.keras.preprocessing.text import tokenizer_from_json
tokenizer = tokenizer_from_json(tokenizer_json)

label_encoder_filename = 'label_encoder.npy'
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(label_encoder_filename, allow_pickle=True)

num_texts = int(input("Enter the number of cipher texts you want to test : "))
for i in range(num_texts):
    user_input = input("Enter cipher text {} for testing : ".format(i + 1))
    user_input = user_input.lower() 
    max_sequence_length = 40
    user_input_tokenized = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_tokenized, maxlen=max_sequence_length)
    prediction_one_hot = loaded_model.predict(user_input_padded)
    predicted_label_numeric = np.argmax(prediction_one_hot, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_label_numeric)[0]
    print("Predicted Label for input {} : {}".format(i + 1, predicted_label))
