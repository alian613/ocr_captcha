import numpy as np
import torch


def decode_prediction(output_tensor, character_set, char_length):
    pred_str = ''
    all_char_set_len = len(character_set)
    for i in range(char_length):
        slice = output_tensor[i * all_char_set_len: (i + 1) * all_char_set_len]
        pred_char = character_set[torch.argmax(slice).item()]
        pred_str += pred_char
    return pred_str


def decode(vec, character_set):
    all_char_set_len = len(character_set)
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % all_char_set_len
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


def encode(text, character_set, character_length):
    all_char_set_len = len(character_set)
    vector = np.zeros(all_char_set_len * character_length, dtype=float)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 65 + 10
            if k > 35:
                k = ord(c) - 97 + 26 + 10
                if k > 61:
                    raise ValueError('error')
        return k

    all_character_set_len = len(character_set)
    for i, c in enumerate(text):
        idx = i * all_character_set_len + char2pos(c)
        vector[idx] = 1.0
    return vector
