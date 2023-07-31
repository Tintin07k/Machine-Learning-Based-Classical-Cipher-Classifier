import random
import nltk
import pandas as pd
nltk.download('words')
from nltk.corpus import words

word_list = random.sample([word for word in words.words() if len(word) >= 4], 10000)

def vigenere_encrypt(plain_text, key):
    
    cipher_text = ""
    key_index = 0
    for char in plain_text:
        if char.isalpha():
            char = char.upper()
            key_char = key[key_index % len(key)].upper()
            encrypted_char = chr(((ord(char) + ord(key_char) - 2 * ord('A')) % 26) + ord('A'))
            cipher_text += encrypted_char
            key_index += 1
        else:
            cipher_text += char
    return cipher_text

def vigenere_decrypt(cipher_text, key):
    
    plain_text = ""
    key_index = 0
    for char in cipher_text:
        if char.isalpha():
            char = char.upper()
            key_char = key[key_index % len(key)].upper()
            decrypted_char = chr(((ord(char) - ord(key_char) + 26) % 26) + ord('A'))
            plain_text += decrypted_char
            key_index += 1
        else:
            plain_text += char
    return plain_text

key = 'KEY'
encrypted_words = [vigenere_encrypt(word, key) for word in word_list]

decrypted_words = [vigenere_decrypt(word, key) for word in encrypted_words]

df = pd.DataFrame({'Plain Text': word_list, 'Key': key, 'Cipher Text': encrypted_words, 'Decrypted Text': decrypted_words})

df.to_excel('VIGENERE_CIPHER_DATASET.xlsx', index=False)
