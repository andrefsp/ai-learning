import numpy as np


FILL_CHAR = '-'

UNKNOWN_CHAR = '#'

ALPHABET = [chr(c) for c in range(ord('a'), ord('z') + 1)] + [
    UNKNOWN_CHAR, FILL_CHAR
]


def _preprocess(string, input_length=100):
    alphabet_dict = {char: i for i, char in enumerate(ALPHABET)}
    encoded = [
        alphabet_dict.get(char, alphabet_dict[UNKNOWN_CHAR])
        for char in string
    ]
    padding = [alphabet_dict[FILL_CHAR]] * (input_length - len(string))
    return np.array(encoded + padding)


def preprocess(strings, input_length=100):
    if isinstance(strings, str):
        return _preprocess(strings, input_length=input_length).reshape(1, input_length)

    return np.array([
        _preprocess(string, input_length=input_length)
        for string in strings
    ])
