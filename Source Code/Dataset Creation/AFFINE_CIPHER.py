import pandas as pd
import random
import nltk
nltk.download('words')
from nltk.corpus import words

class AffineCipher:
    
    def __init__(self, a, b):
        
        self.a = a
        self.b = b
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
    def encrypt(self, message):
        
        encrypted_message = ''
        for letter in message:
            if letter.isalpha():
                letter_index = self.alphabet.index(letter.upper())
                encrypted_index = (self.a * letter_index + self.b) % 26
                encrypted_letter = self.alphabet[encrypted_index]
                encrypted_message += encrypted_letter
            else:
                encrypted_message += letter
        return encrypted_message
    
    def decrypt(self, message):
        
        decrypted_message = ''
        a_inverse = self.mod_inverse(self.a, 26)
        for letter in message:
            if letter.isalpha():
                letter_index = self.alphabet.index(letter.upper())
                decrypted_index = (a_inverse * (letter_index - self.b)) % 26
                decrypted_letter = self.alphabet[decrypted_index]
                decrypted_message += decrypted_letter
            else:
                decrypted_message += letter
        return decrypted_message
    
    def mod_inverse(self, a, m):
        
        for x in range(1, m):
            if (a * x) % m == 1:
                return x
        return -1


random.seed(42) 
nltk_words = words.words()
plaintexts = random.choices(nltk_words, k=10000)

a = 5
b = 8

cipher = AffineCipher(a, b)

encrypted_texts = [cipher.encrypt(plaintext) for plaintext in plaintexts]
decrypted_texts = [cipher.decrypt(encrypted_text) for encrypted_text in encrypted_texts]

data = {'Plain Text': plaintexts, 'Key - a': a, 'Key - b': b, 'Cipher Text': encrypted_texts, 'Decrypted Text': decrypted_texts}
df = pd.DataFrame(data)

df.to_excel('AFFINE_CIPHER_DATASET.xlsx', index=False)
