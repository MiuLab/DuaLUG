import os
from text_token import _UNK, _PAD, _BOS, _EOS
import numpy as np
import torch
from torch import optim


# collate function for NLG
def collate_fn_nlg(batch):
    '''
    n_layers = len(batch[0][1])
    is_inter = False
    if is_inter:
        encoder_input, decoder_labels, inter_labels = \
            [], [[] for _ in range(n_layers)], [[] for _ in range(n_layers)]
    else:
        encoder_input, decoder_labels, refs, sf_data  = [], [[] for _ in range(n_layers)], [], []
    for data in batch:
        encoder_input.append(data[0])
        for idx in range(n_layers):
            decoder_labels[idx].append(data[1][idx])
            if is_inter:
                inter_labels[idx].append(data[2][idx])

    en_max_length = max([len(sent) for sent in encoder_input])
    de_max_lengths = [
            max([len(sent) for sent in labels]) for labels in decoder_labels]
    de_lengths = [
            sum(len(sent) for sent in labels) for labels in decoder_labels]

    encoder_input = pad_sequences(encoder_input, en_max_length, 'pre')
    for idx in range(n_layers):
        decoder_labels[idx] = \
                pad_sequences(decoder_labels[idx], de_max_lengths[idx], 'post')

    for data in batch:
        refs.append(pad_sequences(data[2], de_max_lengths[-1], 'post'))
        sf_data.append(data[3])

    if is_inter:
        for idx in range(n_layers):
            inter_labels[idx] = \
                pad_sequences(inter_labels[idx], de_max_lengths[idx], 'post')
        return encoder_input, decoder_labels, inter_labels, de_lengths
    else:
        # return encoder_input, decoder_labels, de_lengths
        return encoder_input, decoder_labels, de_lengths, refs, sf_data
    '''
    encoder_input, decoder_label, refs, sf_data  = [], [], [], []
    for data in batch:
        encoder_input.append(data[0])
        decoder_label.append(data[1])
    # en_max_length = max([len(sent) for sent in encoder_input])
    de_max_length = max([len(sent) for sent in decoder_label])
    # de_length = sum(len(sent) for sent in decoder_label)
    # encoder_input = pad_sequences(encoder_input, en_max_length, 'pre')
    decoder_label = pad_sequences(decoder_label, de_max_length, 'post')
    for data in batch:
        refs.append(pad_sequences(data[2], de_max_length, 'post'))
        sf_data.append(data[3])
    return encoder_input, decoder_label, refs, sf_data


# collate function for NLU
def collate_fn_nlu(batch):
    encoder_input, decoder_label, refs, sf_data  = [], [], [], []
    for data in batch:
        encoder_input.append(data[1])
        decoder_label.append(data[0])
    en_max_length = max([len(sent) for sent in encoder_input])
    # de_max_length = max([len(sent) for sent in decoder_label])
    # de_length = sum(len(sent) for sent in decoder_label)
    encoder_input = pad_sequences(encoder_input, en_max_length, 'pre')
    # decoder_label = pad_sequences(decoder_label, de_max_length, 'post')
    for data in batch:
        refs.append(pad_sequences(data[2], en_max_length, 'pre'))
        sf_data.append(data[3])
    return encoder_input, decoder_label, refs, sf_data


# collated function for NL
def collate_fn_nl(batch):
    encoder_input, decoder_label, refs, sf_data  = [], [], [], []
    for data in batch:
        encoder_input.append([_BOS] + data[1][:-1])
        decoder_label.append(data[1])
    en_max_length = max([len(sent) for sent in encoder_input])
    de_max_length = max([len(sent) for sent in decoder_label])
    assert en_max_length == de_max_length
    de_length = sum(len(sent) for sent in decoder_label)
    encoder_input = pad_sequences(encoder_input, en_max_length, 'post')
    decoder_label = pad_sequences(decoder_label, de_max_length, 'post')
    for data in batch:
        refs.append(pad_sequences(data[2], de_max_length, 'post'))
        sf_data.append(data[3])
    return encoder_input, decoder_label, refs, sf_data


# collated function for SF
def collate_fn_sf(batch):
    '''
    Warning: this function is not correctly implemented
    '''
    encoder_input, decoder_label, refs, sf_data  = [], [], [], []
    for data in batch:
        encoder_input.append(data[0])
        decoder_labels.append(data[0])
    en_max_length = max([len(sent) for sent in encoder_input])
    de_max_length = max([len(sent) for sent in decoder_label])
    de_length = sum(len(sent) for sent in decoder_label)
    encoder_input = pad_sequences(encoder_input, en_max_length, 'pre')
    decoder_label = pad_sequences(decoder_label, de_max_length, 'post')
    for data in batch:
        refs.append(pad_sequences(data[2], de_max_length, 'post'))
        sf_data.append(data[3])
    return encoder_input, decoder_label, refs, sf_data


def pad_sequences(data, max_length, pad_type):
    # print(data)
    if _PAD != -1:
        padded_data = np.full((len(data), max_length), _PAD)
    else:
        padded_data = np.full((len(data), max_length), _UNK)
    if pad_type == "post":
        for idx, d in enumerate(data):
            padded_data[idx][:min(max_length, len(d))] = \
                    d[:min(max_length, len(d))]
    elif pad_type == "pre":
        for idx, d in enumerate(data):
            padded_data[idx][max(0, max_length-len(d)):] = \
                    d[:min(max_length, len(d))]
    return padded_data


# Model helper
def get_embeddings(vocab, embeddings_dir, embedding_dim):
    embedding_file = os.path.join(
            embeddings_dir, "glove.6B.{}d.txt".format(embedding_dim))
    embeddings = torch.nn.Parameter(
            torch.Tensor(torch.randn(len(vocab), embedding_dim)))
    with open(embedding_file, 'r') as file:
        for line in file:
            data = line.strip().split(' ')
            word, emb = \
                data[0], torch.Tensor(np.array(list(map(float, data[1:]))))
            if word not in vocab:
                continue
            embeddings.data[vocab[word]] = torch.Tensor(emb)

    return embeddings


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_optimizer(optimizer, parameters, learning_rate):
    if optimizer == "Adam":
        return optim.Adam(
                parameters, lr=learning_rate)
    elif optimizer == "RMSprop":
        return optim.RMSprop(
                parameters, lr=learning_rate)
    elif optimizer == "SGD":
        return optim.SGD(
                parameters, lr=learning_rate)
