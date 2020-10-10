import pickle
import os
import argparse

import numpy as np


def sequences_to_nhot(seqs, vocab_size):
    """
    args:
        seqs: list of list of word_ids
        vocab_size: int

    outputs:
        labels: np.array of shape [batch_size, vocab_size]
    """
    labels = np.zeros((len(seqs), vocab_size), dtype=np.float32)
    for bid, seq in enumerate(seqs):
        for word in seq:
            labels[bid][word] = 1
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data')
    parser.add_argument('valid_data')
    parser.add_argument('vocab_file')
    parser.add_argument('save_path')
    args = parser.parse_args()

    _, _, token_vocab, _ = pickle.load(open(args.vocab_file, 'rb'))
    attr_vocab_size = len(token_vocab)

    train_data, _, _, _, _ = pickle.load(open(args.train_data, 'rb'))
    valid_data, _, _, _, _ = pickle.load(open(args.valid_data, 'rb'))

    train_nhot = sequences_to_nhot(train_data, attr_vocab_size)
    valid_nhot = sequences_to_nhot(valid_data, attr_vocab_size)

    np.savez(args.save_path, train_data=train_nhot, valid_data=valid_nhot)
