from nltk.corpus import wordnet, treebank
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import spacy
import os
import ast
import random
import pickle
import operator
from utils import print_time_info
from text_token import _UNK, _PAD, _BOS, _EOS


class Tokenizer:
    def __init__(self, vocab_path, split_vocab, regen, train):
        self.vocab_path = vocab_path
        self.split_vocab = split_vocab
        if (not regen or not train):
            if os.path.exists(vocab_path):
                print_time_info(
                        "Read vocab data from {}".format(self.vocab_path))
                if self.split_vocab:
                    self.vocab, self.rev_vocab, \
                        self.token_vocab, self.rev_token_vocab = \
                        pickle.load(open(self.vocab_path, 'rb'))
                else:
                    self.vocab, self.rev_vocab = \
                        pickle.load(open(self.vocab_path, 'rb'))
            else:
                print_time_info("Vocab file doesn't exist...")

    def build_vocab(self, corpus, tokens=None):
        # You should pass a list with all words in the dataset as corpus
        self.vocab, self.rev_vocab = {}, []
        self.vocab['_UNK'] = len(self.rev_vocab)
        self.rev_vocab.append('_UNK')
        self.vocab['_PAD'] = len(self.rev_vocab)
        self.rev_vocab.append('_PAD')
        self.vocab['_BOS'] = len(self.rev_vocab)
        self.rev_vocab.append('_BOS')
        self.vocab['_EOS'] = len(self.rev_vocab)
        self.rev_vocab.append('_EOS')
        print_time_info(
                "Build vocab: {} words".format(len(corpus)))
        raw_vocab = {}
        for word in corpus:
            if word not in raw_vocab:
                raw_vocab[word] = 0
            raw_vocab[word] += 1

        sorted_vocab = sorted(
                raw_vocab.items(), key=operator.itemgetter(1))[::-1]
        word_cnt = 0
        for idx, word in enumerate(sorted_vocab):
            word_cnt += word[1]
            if ((word_cnt / len(corpus)) >= 0.9
                    and (word_cnt - word[1]) / len(corpus) < 0.9):
                print_time_info("90% coverage: vocab size {}".format(idx))
            if ((word_cnt / len(corpus)) >= 0.95
                    and ((word_cnt - word[1]) / len(corpus)) < 0.95):
                print_time_info("95% coverage: vocab size {}".format(idx))
            if ((word_cnt / len(corpus)) >= 0.99
                    and ((word_cnt - word[1]) / len(corpus)) < 0.99):
                print_time_info("99% coverage: vocab size {}".format(idx))
        print_time_info(
                "100% coverage: vocab size {}".format(len(sorted_vocab)))

        for word, _ in sorted_vocab:
            self.vocab[word] = len(self.rev_vocab)
            self.rev_vocab.append(word)

        if self.split_vocab:
            self.token_vocab, self.rev_token_vocab = {}, []
            '''
            self.token_vocab['_UNK'] = len(self.rev_token_vocab)
            self.rev_token_vocab.append('_UNK')
            self.token_vocab['_PAD'] = len(self.rev_token_vocab)
            self.rev_token_vocab.append('_PAD')
            '''
            raw_vocab = {}
            for token in tokens:
                if token not in raw_vocab:
                    raw_vocab[token] = 0
                raw_vocab[token] += 1

            sorted_vocab = sorted(
                    raw_vocab.items(), key=operator.itemgetter(1))[::-1]

            for token, _ in sorted_vocab:
                self.token_vocab[token] = len(self.rev_token_vocab)
                self.rev_token_vocab.append(token)

        print_time_info("Save vocab data to {}".format(self.vocab_path))
        if not tokens:
            pickle.dump(
                    [self.vocab, self.rev_vocab], open(self.vocab_path, 'wb'))
        else:
            pickle.dump(
                    [
                        self.vocab, self.rev_vocab,
                        self.token_vocab, self.rev_token_vocab
                    ], open(self.vocab_path, 'wb'))

    def shrink_vocab(self, vocab_size):
        special_token = " + _UNK, _BOS, _EOS, _PAD"
        print_time_info(
                "Shrink vocab size to {}{}".format(vocab_size, special_token))
        # 4 for special token
        shrink_rev_vocab = self.rev_vocab[vocab_size + 4:]
        for word in shrink_rev_vocab:
            self.vocab.pop(word)
        self.rev_vocab = self.rev_vocab[:vocab_size + 4]

    def tokenize(self, sent, is_token):
        token_sent = []
        if is_token and self.split_vocab:
            for word in sent:
                token_sent.append(self.token_vocab[word])
        else:
            for word in sent:
                if word in self.vocab:
                    token_sent.append(self.vocab[word])
                else:
                    token_sent.append(_UNK)
        return token_sent

    def untokenize(self, sent, sf_data=None, is_token=False):
        untoken_sent = []
        if is_token and self.split_vocab:
            for token in sent:
                token = int(token)
                untoken_sent.append(self.rev_token_vocab[token])
        else:
            for token in sent:
                token = int(token)
                # if token is _EOS: break
                # if token in [_BOS, _PAD]: continue
                untoken_sent.append(self.rev_vocab[token])
        '''
        untoken_sent = [sf_data['name'] if ('name' in sf_data) and (t == 'NAME' or t == 'NAMETOKEN') else t for t in untoken_sent]
        untoken_sent = [sf_data['near'] if ('near' in sf_data) and (t == 'NEAR' or t == 'NEARTOKEN') else t for t in untoken_sent]
        '''
        return untoken_sent
