# Machine Learning Based Classical Cipher Classifier

## Overview

<p align="justify">This project aims to build a machine learning based classifier to distinguish between various classical ciphers. Classical ciphers are historical encryption techniques that perform simple substitution or transposition on plaintext to produce ciphertext. By training a machine learning model on labeled examples of ciphertexts from different ciphers, the classifier can predict the cipher used to encrypt an unseen ciphertext.</p>

## Classical Ciphers

### Group 1 : Substitution Ciphers

#### 1. Playfair Cipher

#### 2. Polybius Square Cipher

#### 3. Baconian Cipher

#### 4. Atbash Cipher

### Group 2 : Transposition Ciphers

#### 5. Columnar Transposition Cipher

#### 6. Scytale Cipher

#### 7. Rail Fence Cipher

#### 8. Caesar Cipher

### Group 3 : Advanced Ciphers

#### 9. Hill Cipher

#### 10. Affine Cipher

#### 11. Vigenère Cipher

#### 12. Gronsfeld Cipher

## Dataset 

### Dataset Preparation 

<p align="justify">The dataset was prepared using the Natural Language Toolkit ( NLTK ) module in Python. NLTK is a powerful library for natural language processing and text analysis. We used NLTK to process text data and create a customized dataset for our project.</p>

<p align="justify">In addition to using NLTK to process text data , we needed to augment our dataset with random words of random length. For this purpose , we created Python code that generates random words.</p>

### Dataset Information

<p align="justify">Each cipher dataset consists of 1000 plaintext samples, encryption keys, ciphertext (encrypted plaintext), and decrypted plaintext.</p>

#### Dataset Components

<p align="justify">The cipher dataset is organized into the following components :</p>

1. **Plaintext** : <p align="justify">This dataset contains 1000 plaintext samples. Plaintext refers to the original, unencrypted data or messages that are used as input for encryption.</p>

2. **Key for Encryption** : <p align="justify">For each plaintext sample, there is a corresponding encryption key. The encryption key is a parameter used by the encryption algorithm to transform the plaintext into ciphertext.</p>

3. **Decrypted Plaintext** : <p align="justify">Along with the plaintext and encryption key, the dataset includes the decrypted plaintext. After applying decryption using the corresponding key, the ciphertext should be transformed back into its original form, which is the decrypted plaintext.</p>

4. **Ciphertext ( Encrypted Plaintext )** : <p align="justify">For each plaintext sample, there is a corresponding ciphertext. Ciphertext is the result of applying encryption to the plaintext using the encryption key. It represents the encrypted version of the original plaintext.</p>

## Machine Learning Model

### LSTM ( Long Short - Term Memory ) RNN

<p align="justify">LSTM is a type of recurrent neural network architecture designed to overcome the limitations of traditional RNNs ( Recurrent Neural Network ) in handling long-term dependencies. It introduces memory cells that allow the model to remember information over extended sequences. The key components of an LSTM cell are :</p>

1. **Input Gate** : <p align="justify">Controls how much new information is added to the cell's state.</p>
2. **Forget Gate** : <p align="justify">Controls what information to discard from the previous state.</p>
3. **Output Gate** : <p align="justify">Controls how much information from the cell's state is used to produce the output.</p>

<p align="justify">LSTM networks are well - suited for tasks where preserving context information over long sequences is critical, such as language modeling, speech recognition, and sentiment analysis.</p>

### Bidirectional LSTM RNN

<p align="justify">A Bidirectional LSTM RNN is an extension of the standard LSTM, which processes the input data in both forward and backward directions. By allowing the model to access future context information during training , Bidirectional LSTM can capture patterns from both past and future timesteps , resulting in enhanced performance for sequence modeling tasks.</p>

<p align="justify">The Bidirectional LSTM combines two LSTM layers, one processing the input sequence forward and the other backward. The outputs of both LSTM layers are then concatenated , creating a comprehensive representation of the input sequence.</p>

<p align="justify">Bidirectional LSTM RNNs are particularly effective in tasks where future context is essential , such as speech recognition , named entity recognition and machine translation.</p>

<p align="justify">In the context of using Bidirectional LSTM, we have implemented the various combinations of units_lstm ( the number of units in the LSTM layers ) and dropout rate to find the best model for our specific task. The process will involve training multiple Bidirectional LSTM models with different units_lstm and dropout rate values , and then evaluating their performance on a validation set.</p>

<p align="justify">To perform this hyperparameter search, we may use techniques such as grid search, random search, or more advanced optimization methods like Bayesian optimization. The best model will be selected based on its performance metrics , such as accuracy , precision or recall.</p>

### Usage

<p align="justify">In our project , we have leveraged the popular deep learning frameworks such as TensorFlow , Keras or PyTorch to implement LSTM and Bidirectional LSTM models. These frameworks offer user - friendly APIs and pre - built layers for creating these recurrent neural network architectures.</p>

<p align="justify">Before feeding the sequence data into the LSTM and Bidirectional LSTM models , it's essential to preprocess the data appropriately. Preprocessing steps may include tokenization , padding and converting the text data into numerical representations suitable for training the models.</p>

## Testing of the Model

<p align="justify">During the testing process, we employed a trained machine learning model along with a tokenizer configuration and label encoder file. These components are crucial for accurately evaluating the performance of the model on unseen data.</p>

### Trained Model

<p align="justify">The trained machine learning model serves as the core component for decryption during testing. It was previously trained on the training data, where it learned the underlying patterns and relationships in the ciphertext. Using its knowledge from the training phase , the model can predict the corresponding plaintext when provided with encrypted data.</p>

### Tokenizer Configuration

<p align="justify">The tokenizer configuration is utilized to preprocess the raw text data , converting it into a format suitable for the model's input. During training, the same tokenizer configuration was used to process the training data , ensuring consistency between training and testing. The tokenizer handles tasks such as tokenization , padding and numerical conversion , enabling the model to comprehend the input data effectively.</p>

### Label Encoder File

<p align="justify">In cryptography tasks , especially when dealing with classification based ciphers , a label encoder file is employed to encode the categorical labels of the ciphertext. This encoding facilitates the comparison of the model's predictions with the actual plaintext during testing , enabling the calculation of accuracy and other performance metrics.</p>

### Evaluation Process

<p align="justify">During the testing phase, we feed the encrypted ciphertext into the trained model , which employs the provided tokenizer configuration to preprocess the data. The model then utilizes its learned parameters to make predictions, generating the corresponding plaintext. The predicted plaintext is then compared to the actual plaintext ( encoded using the label encoder file ) to assess the model's accuracy and performance.</p>

<p align="justify">By using a trained model , tokenizer configuration and label encoder file during testing , we ensure that the evaluation is consistent with the training process, yielding reliable insights into the model's ability to accurately decrypt ciphertext and handle various ciphers.</p>

## Model Metrics Evaluation

<p align="justify">Our machine learning model exhibits robust performance across all 3 cipher groups, achieving a consistent accuracy of around 95% for both training and testing data. This high accuracy validates the effectiveness of the model in decrypting various types of ciphers and demonstrates its potential for real-world applications in cryptographic analysis and message decryption.</p>

## Working of Each Cipher

### Group 1 : Substitution Ciphers

#### 1. Playfair Cipher

- **Key :** <p align="justify">The Playfair Cipher requires a keyword to construct the Playfair Square matrix. The keyword should be unique, with no duplicate letters, and is used to generate the matrix.</p>
- **Encryption :** <p align="justify">The plaintext must be processed in pairs of two letters. If there is an odd number of letters, a dummy letter (like 'X') is added at the end. The encryption process involves finding the positions of the pair in the Playfair Square and applying specific rules to determine the ciphertext.</p>
- **Decryption :** <p align="justify">The decryption process is similar to encryption, but it involves reversing the rules used during encryption to find the original plaintext.</p>

#### 2. Polybius Square Cipher

- **Key :** <p align="justify">The Polybius Square Cipher does not use a traditional key like other ciphers. Instead, it relies on the fixed 5x5 matrix (Polybius Square) where each letter of the alphabet corresponds to a unique row and column position.</p>
- **Encryption :** <p align="justify">The plaintext is converted into its numerical representation based on the row and column positions in the Polybius Square. The numerical values are then concatenated to form the ciphertext.</p>
- **Decryption :** <p align="justify">Decryption requires a knowledge of the Polybius Square to reverse the process and find the original plaintext.</p>

#### 3. Baconian Cipher

- **Key :** <p align="justify">The Baconian Cipher uses a fixed 5-letter alphabet (A and B) to represent each letter of the English alphabet. There is no additional key required for the encryption or decryption process.</p>
- **Encryption :** <p align="justify">Each letter in the plaintext is converted into its corresponding Baconian representation based on the 5-letter alphabet. The resulting binary sequence forms the ciphertext.</p>
- **Decryption :** <p align="justify">Decryption involves reversing the process by converting the binary sequence back to plaintext based on the Baconian alphabet.</p>

#### 4. Atbash Cipher

- **Key :** <p align="justify">The Atbash Cipher does not require a key for encryption or decryption. The substitution pattern is fixed and remains the same for all messages.</p>
- **Encryption :** <p align="justify">Each letter in the plaintext is replaced by its reverse in the alphabet. For example, 'A' becomes 'Z,' 'B' becomes 'Y,' and so on.</p>
- **Decryption :** <p align="justify">The decryption process is the same as encryption since the substitution pattern is fixed.</p>

### Group 2 : Transposition Ciphers

#### 5. Columnar Transposition Cipher

- **Key :** <p align="justify">The Columnar Transposition Cipher requires a keyword to determine the column order during encryption and decryption. The keyword should contain unique letters with no duplicates.</p>
- **Encryption :** <p align="justify">The plaintext is written in rows under the keyword, and then the columns are rearranged based on the alphabetical order of the letters in the keyword. The ciphertext is read column by column to generate the encrypted message.</p>
- **Decryption :** <p align="justify">Decryption involves rearranging the columns based on the original keyword to obtain the original plaintext.</p>

#### 6. Scytale Cipher

- **Key :** <p align="justify">The Scytale Cipher uses a cylindrical device called a "scytale" as its key. The scytale has a specific circumference that determines the number of letters to be written on each turn during encryption and decryption.</p>
- **Encryption :** <p align="justify">The plaintext is written lengthwise along the scytale, and the ciphertext is read off the cylinder in a spiral pattern to produce the encrypted message.</p>
- **Decryption :** <p align="justify">To decrypt, the scytale must have the same circumference as used during encryption. The ciphertext is wrapped around the scytale, and the plaintext is read off in a straight line to reveal the original message.</p>

#### 7. Rail Fence Cipher

- **Key :** <p align="justify">The Rail Fence Cipher requires the number of "rails" or rows as its key, which determines the height of the zigzag pattern.</p>
- **Encryption :** <p align="justify">The plaintext is written diagonally along the rails, and then the ciphertext is read off row by row to produce the encrypted message.</p>
- **Decryption :** <p align="justify">Decryption involves recreating the zigzag pattern by placing the ciphertext letters in the appropriate rails, and then reading the plaintext off diagonally.</p>

#### 8. Caesar Cipher

- **Key :** <p align="justify">The Caesar Cipher requires a fixed integer value (key) that determines the shift for encryption and decryption. The key value represents the number of positions each letter is shifted down the alphabet.</p>
- **Encryption :** <p align="justify">Each letter in the plaintext is shifted down the alphabet by the key value to produce the ciphertext. For example, with a shift of 3, 'A' becomes 'D,' 'B' becomes 'E,' and so on.</p>
- **Decryption :** <p align="justify">Decryption involves shifting each letter in the ciphertext up the alphabet by the negative key value to reveal the original plaintext.</p>

### Group 3 : Advanced Ciphers

#### 9. Hill Cipher

- **Key :** <p align="justify">The Hill Cipher requires a square matrix called the encryption key matrix. The matrix must be invertible to allow for decryption.</p>
- **Encryption :** <p align="justify">The plaintext is divided into blocks of letters, each represented as a matrix. Encryption involves multiplying each matrix with the encryption key matrix to produce the ciphertext matrix.</p>
- **Decryption :** <p align="justify">To decrypt, the ciphertext matrix is multiplied by the inverse of the encryption key matrix to obtain the original plaintext matrix, which is then converted back to text.</p>

#### 10. Affine Cipher

- **Key :** <p align="justify">The Affine Cipher requires two numeric keys, 'a' and 'b,' with 'a' being coprime with 26 (the number of letters in the English alphabet). The key 'a' determines the multiplication factor, and 'b' determines the shift value for encryption and decryption.</p>
- **Encryption :** <p align="justify">Each letter in the plaintext is first converted to a numerical value. Encryption involves applying the mathematical operation: ciphertext = (a * plaintext + b) % 26, where '%' represents the modulo operation.</p>
- **Decryption :** <p align="justify">Decryption requires finding the modular inverse of 'a,' which allows for the reverse operation to find the original plaintext.</p>

#### 11. Vigenère Cipher

- **Key :** <p align="justify">The Vigenère Cipher uses a keyword that is repeated as many times as needed to match the length of the plaintext. The letters of the keyword determine the letter shift for each corresponding letter in the plaintext.</p>
- **Encryption :** <p align="justify">Each letter in the plaintext is shifted based on the corresponding letter in the keyword to generate the ciphertext.</p>
- **Decryption :** <p align="justify">Decryption involves finding the letters of the keyword needed to reverse the shifts and obtain the original plaintext.</p>

#### 12. Gronsfeld Cipher

- **Key :** <p align="justify">The Gronsfeld Cipher requires a numeric key, similar to the Vigenère Cipher. The key is repeated as needed to match the length of the plaintext, and each numeric value determines the letter shift for the corresponding letter in the plaintext.</p>
- **Encryption :** <p align="justify">Each letter in the plaintext is shifted based on the corresponding numeric value in the key to generate the ciphertext.</p>
- **Decryption :** <p align="justify">Decryption involves finding the numeric values of the key needed to reverse the shifts and obtain the original plaintext.</p>

## Conclusion

<p align="justify">Leveraging machine learning to train a classifier on labeled datasets of ciphertexts encrypted with classical ciphers offers a powerful tool for identifying the encryption technique applied to unseen ciphertexts. This classifier can play a vital role in analyzing and deciphering encrypted messages, thereby offering valuable insights into historical encryption methods and their respective strengths in ensuring security. Understanding classical ciphers not only allows us to marvel at the brilliance of historical cryptographers but also lays a strong foundation for comprehending and developing modern cryptographic techniques that are essential for safeguarding computer security and protecting sensitive data in today's digital age.</p>
