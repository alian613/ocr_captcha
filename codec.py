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
    for c in char_pos:
        char_idx = c % all_char_set_len
        text.append(character_set[char_idx])
    return "".join(text)



def encode(text, character_set, character_length):
    all_char_set_len = len(character_set)
    vector = np.zeros(all_char_set_len * character_length, dtype=float)

    for i, c in enumerate(text):
        if c not in character_set:
            raise ValueError(f"Invalid character '{c}' not in character set.")
        idx = i * all_char_set_len + character_set.index(c)
        vector[idx] = 1.0

    return vector
