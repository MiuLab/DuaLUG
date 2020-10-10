'''
Do the preprocessing of e2e dataset
'''
import pandas as pd
from os.path import join as opj
from tqdm import tqdm

# nltk toolkit
import nltk
import spacy

# Multiprocess
from multiprocessing import Pool as ProcessPool
import multiprocessing

spacy_nlp = spacy.load('en')


def nltk_pos(sent):
    text = nltk.word_tokenize(sent, tagset='universal')
    pos_tag = nltk.pos_tag(text)
    words, tags = [], []
    for w, t in pos_tag:
        words.append(w)
        if t != '.':
            tags.append(t)
        else:
            tags.append('PUNCT')
    return words, tags


def spacy_pos(sent):
    text = spacy_nlp(sent)
    words, tags = [], []
    for word in text:
        words.append(word.text)
        tags.append(word.pos_)
    return words, tags


def both_pos(sent, tolerence=0):
    nltk_word, nltk_tag = nltk_pos(sent)
    spacy_word, spacy_tag = spacy_pos(sent)

    success = False
    if len(set(nltk_tag) - set(spacy_tag)) <= tolerence:
        success = True

    return spacy_word, spacy_tag, success


class Preprocessor:
    def __init__(self, savedir, tokenizer):
        self.savedir = savedir

        if tokenizer == 'spacy':
            self.tokenizer = spacy_pos
        elif tokenizer == 'nltk':
            self.tokenizer = nltk_pos
        else:
            # default tokenizer = spacy
            self.tokenizer = spacy_pos

    def e2e(self, data_dir):
        # On going
        train = pd.read_csv(opj(data_dir, 'trainset.csv'))

        train_sent = train['ref'].values

        cpus = multiprocessing.cpu_count()
        workers = ProcessPool(cpus)

        for words, tags in tqdm(workers.imap(self.tokenizer, train_sent),
                                    desc='E2E'):
            l1_sent, l2_sent, full = [], [], []
            for word, tag in zip(words, tags):
                full.append(word)
                if tag in ['NOUN', 'PROPN', 'PRON']:
                    l1_sent.append(word)
                    l2_sent.append(word)
                if tag == 'VERB':
                    l2_sent.append(word)
