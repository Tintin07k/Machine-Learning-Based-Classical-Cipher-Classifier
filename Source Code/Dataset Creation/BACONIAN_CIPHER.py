import pandas as pd
import random
import nltk
from nltk.corpus import words
nltk.download('words')
word_list = words.words()

random_words = random.sample(word_list, k=10000)

baconian_alphabet = {
    'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB', 'E': 'AABAA',
    'F': 'AABAB', 'G': 'AABBA', 'H': 'AABBB', 'I': 'ABAAA', 'J': 'ABAAA',
    'K': 'ABAAB', 'L': 'ABABA', 'M': 'ABABB', 'N': 'ABBAA', 'O': 'ABBAB',
    'P': 'ABBBA', 'Q': 'ABBBB', 'R': 'BAAAA', 'S': 'BAAAB', 'T': 'BAABA',
    'U': 'BAABB', 'V': 'BAABB', 'W': 'BABAA', 'X': 'BABAB', 'Y': 'BABBA',
    'Z': 'BABBB'
}

def baconian_encode(message):
    
    encoded_message = ''
    for char in message.upper():
        if char in baconian_alphabet:
            encoded_message += baconian_alphabet[char] + ' '
        else:
            encoded_message += char
    return encoded_message.strip()

def baconian_decode(encoded_message):
    
    decoded_message = ''
    binary_chars = ''
    for char in encoded_message:
        if char == ' ':
            if binary_chars:
                for letter, binary in baconian_alphabet.items():
                    if binary == binary_chars:
                        decoded_message += letter
                        break
                binary_chars = ''
        else:
            binary_chars += char

    if binary_chars:
        for letter, binary in baconian_alphabet.items():
            if binary == binary_chars:
                decoded_message += letter
                break

    return decoded_message

results = []

for word in random_words:
    encrypted_word = baconian_encode(word)
    decrypted_word = baconian_decode(encrypted_word)
    results.append((word, encrypted_word, decrypted_word))

df = pd.DataFrame(results, columns=['Plain Text', 'Cipher Text', 'Decrypted Text'])

df.to_excel('BACONIAN_CIPHER_DATASET.xlsx', index=False)
