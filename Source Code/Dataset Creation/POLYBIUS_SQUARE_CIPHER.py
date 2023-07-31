import string
import random
import nltk
import pandas as pd
nltk.download('words')

def create_polybius_square():
    
    square = [
        ['A', 'B', 'C', 'D', 'E'],
        ['F', 'G', 'H', 'I/J', 'K'],
        ['L', 'M', 'N', 'O', 'P'],
        ['Q', 'R', 'S', 'T', 'U'],
        ['V', 'W', 'X', 'Y', 'Z']
    ]
    print("Polybius Square Table:")
    for row in square:
        print(' '.join(row))
    return square

def find_char(square, target):
    
    if target == 'J':
        target = 'I'
    for i in range(5):
        for j in range(5):
            if square[i][j].startswith(target):
                return i, j
    return None, None

def encrypt(plain_text, square):
    
    cipher_text = ''
    for char in plain_text.upper():
        if char.isalpha() or char.isdigit():
            i, j = find_char(square, char)
            if i is not None and j is not None:
                cipher_text += str(i + 1) + str(j + 1)
        else:
            cipher_text += char
    return cipher_text

def decrypt(cipher_text, square):
    
    plain_text = ''
    digits = iter(cipher_text)
    for i, j in zip(digits, digits):
        if i.isdigit() and j.isdigit():
            i, j = int(i) - 1, int(j) - 1
            if 0 <= i < 5 and 0 <= j < 5:
                letter = square[i][j]
                if letter == 'I/J':
                    letter = 'J' if 'I' in cipher_text else 'I'
                plain_text += letter
        else:
            plain_text += i + j
    return plain_text

polybius_square = create_polybius_square()

nltk_words = nltk.corpus.words.words()

wordlist = random.choices(nltk_words, k=10000)

data = {
    'Plain Text': [],
    'Cipher Text': [],
    'Decrypted Text': []
}

for word in wordlist:
    encrypted_word = encrypt(word, polybius_square)
    decrypted_word = decrypt(encrypted_word, polybius_square)
    data['Plain Text'].append(word)
    data['Cipher Text'].append(encrypted_word)
    data['Decrypted Text'].append(decrypted_word)

df = pd.DataFrame(data)

df.to_excel('POLYBIUS_SQUARE_CIPHER_DATASET.xlsx', index=False)
