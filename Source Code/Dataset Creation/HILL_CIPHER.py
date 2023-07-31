import numpy as np
import pandas as pd
from egcd import egcd

alphabet = "abcdefghijklmnopqrstuvwxyz"

letter_to_index = dict(zip(alphabet, range(len(alphabet))))
index_to_letter = dict(zip(range(len(alphabet)), alphabet))


def matrix_mod_inv(matrix, modulus):
    
    det = int(np.round(np.linalg.det(matrix)))  
    det_inv = egcd(det, modulus)[1] % modulus 
    matrix_modulus_inv = (det_inv * np.round(det * np.linalg.inv(matrix)).astype(int) % modulus)
    return matrix_modulus_inv

def encrypt(message, K):
    
    encrypted = ""
    message_in_numbers = []
    for letter in message:
        message_in_numbers.append(letter_to_index[letter])
    split_P = [message_in_numbers[i : i + int(K.shape[0])] for i in range(0, len(message_in_numbers), int(K.shape[0]))]
    for P in split_P:
        P = np.transpose(np.asarray(P))[:, np.newaxis]
        while P.shape[0] != K.shape[0]:
            P = np.append(P, letter_to_index[" "])[:, np.newaxis]
        numbers = np.dot(K, P) % len(alphabet)
        n = numbers.shape[0]
        for idx in range(n):
            number = int(numbers[idx, 0])
            encrypted += index_to_letter[number]
    return encrypted

def decrypt(cipher, Kinv):
    
    decrypted = ""
    cipher_in_numbers = []
    for letter in cipher:
        cipher_in_numbers.append(letter_to_index[letter])
    split_C = [ cipher_in_numbers[i : i + int(Kinv.shape[0])] for i in range(0, len(cipher_in_numbers), int(Kinv.shape[0]))]
    for C in split_C:
        C = np.transpose(np.asarray(C))[:, np.newaxis]
        numbers = np.dot(Kinv, C) % len(alphabet)
        n = numbers.shape[0]
        for idx in range(n):
            number = int(numbers[idx, 0])
            decrypted += index_to_letter[number]
    return decrypted

def generate_random_message(length):
    
    random_message = ""
    for _ in range(length):
        random_message += np.random.choice(list(alphabet))
    return random_message

def matrix_to_letters(matrix):
    
    letter_matrix = [[index_to_letter[i] for i in row] for row in matrix.tolist()]
    return letter_matrix

def main():
    
    num_messages = 10000
    message_length = 8
    K = np.matrix([[8, 6, 9, 5], [6, 9, 5, 10], [5, 8, 4, 9], [10, 6, 11, 4]])
    K_letters = matrix_to_letters(K)
    Kinv = matrix_mod_inv(K, len(alphabet))
    messages = []
    encrypted_messages = []
    decrypted_messages = []
    keys = []
    for _ in range(num_messages):
        message = generate_random_message(message_length)
        encrypted_message = encrypt(message, K)
        decrypted_message = decrypt(encrypted_message, Kinv)
        messages.append(message)
        encrypted_messages.append(encrypted_message)
        decrypted_messages.append(decrypted_message)
        keys.append(K_letters)
    data = {
        "Plain Text": messages,
        "Key": keys,
        "Cipher Text": encrypted_messages,
        "Decrypted Text": decrypted_messages,
    }
    df = pd.DataFrame(data)
    df.to_excel("HILL_CIPHER_DATASET.xlsx", index=False)

main()
