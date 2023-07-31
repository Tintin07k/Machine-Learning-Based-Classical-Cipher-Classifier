import random
import string
import nltk
import pandas as pd
nltk.download('words')
from nltk.corpus import words

def rail_fence_encrypt(plaintext, rails):
    
    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1
    for char in plaintext:
        fence[rail].append(char)
        rail += direction
        if rail == rails - 1 or rail == 0:
            direction *= -1
    return ''.join([''.join(rail) for rail in fence])

def rail_fence_decrypt(ciphertext, rails):
    
    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1
    for _ in range(len(ciphertext)):
        fence[rail].append(None)
        rail += direction
        if rail == rails - 1 or rail == 0:
            direction *= -1
    index = 0
    for rail in fence:
        for i in range(len(rail)):
            rail[i] = ciphertext[index]
            index += 1
    rail = 0
    direction = 1
    plaintext = []
    for _ in range(len(ciphertext)):
        plaintext.append(fence[rail].pop(0))
        rail += direction
        if rail == rails - 1 or rail == 0:
            direction *= -1
    return ''.join(plaintext)

def generate_random_words(num_words, min_length):
    
    all_words = words.words()
    random_words = []
    while len(random_words) < num_words:
        word = random.choice(all_words)
        if len(word) >= min_length:
            random_words.append(word)
    return random_words


random_words = generate_random_words(10000, 4)

encrypted_texts = []
for word in random_words:
    encrypted_word = rail_fence_encrypt(word, 3)
    encrypted_texts.append(encrypted_word)

decrypted_texts = []
for ciphertext in encrypted_texts:
    decrypted_word = rail_fence_decrypt(ciphertext, 3)
    decrypted_texts.append(decrypted_word)

df = pd.DataFrame({'Plain Text': random_words, 'Key': 3, 'Cipher Text': encrypted_texts, 'Decrypted Text': decrypted_texts})

df.to_excel('RAIL_FENCE_CIPHER_DATASET.xlsx', index=False)
