import random
import string
import pandas as pd
import nltk
nltk.download('words')
from nltk.corpus import words

def atbash_encrypt(plaintext):
    
    alphabet = string.ascii_lowercase
    ciphertext = ""
    for char in plaintext:
        if char.isalpha():
            if char.isupper():
                ciphertext += alphabet[25 - (ord(char.lower()) - ord('a'))].upper()
            else:
                ciphertext += alphabet[25 - (ord(char) - ord('a'))]
        else:
            ciphertext += char
    return ciphertext

def atbash_decrypt(ciphertext):
    
    return atbash_encrypt(ciphertext)  

random_words = random.choices([word for word in words.words() if len(word) >= 4], k=10000)

encrypted_words = [atbash_encrypt(word) for word in random_words]

decrypted_words = [atbash_decrypt(word) for word in encrypted_words]

data = {'Plain Text': random_words, 'Key': 'N/A', 'Cipher Text': encrypted_words, 'Decrypted Text': decrypted_words}
df = pd.DataFrame(data)

df.to_excel('ATBASH_CIPHER_DATASET.xlsx', index=False)
