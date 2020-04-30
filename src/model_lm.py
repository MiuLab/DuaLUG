import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import random
import numpy as np
import os
import math

from module import LMRNN
# from utils import single_BLEU, BLEU, single_ROUGE, ROUGE, best_ROUGE, print_time_info, check_dir, print_curriculum_status
from utils import *
from text_token import _UNK, _PAD, _BOS, _EOS
from model_utils import collate_fn_nl, build_optimizer, get_device
from logger import Logger

from tqdm import tqdm


class LM:
    def __init__(
            self,
            batch_size,
            optimizer,
            learning_rate,
            train_data_engine,
            test_data_engine,
            dim_hidden,
            dim_embedding,
            vocab_size=None,
            n_layers=1,
            model_dir="./model",
            log_dir="./log",
            is_load=True,
            replace_model=True,
            device=None,
            dir_name='test'
    ):

        # Initialize attributes
        self.data_engine = train_data_engine
        self.n_layers = n_layers
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.dim_hidden = dim_hidden
        self.dim_embedding = dim_embedding
        self.vocab_size = vocab_size
        self.dir_name = dir_name

        self.device = get_device(device)

        self.lm = LMRNN(
            dim_embedding=dim_embedding,
            dim_hidden=dim_hidden,
            attr_vocab_size=None,
            vocab_size=vocab_size,
            n_layers=n_layers,
            bidirectional=False
        )

        self.lm.to(self.device)

        self.parameters = filter(
                lambda p: p.requires_grad, self.lm.parameters())
        self.optimizer = build_optimizer(
                optimizer, self.parameters, learning_rate)

        self.model_dir, self.log_dir = handle_model_dirs(
            model_dir, log_dir, dir_name, replace_model, is_load
        )

        if is_load:
            self.load_model(self.model_dir)

        self.train_data_engine = train_data_engine
        self.test_data_engine = test_data_engine
        self.train_data_loader = DataLoader(
                train_data_engine,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=True,
                collate_fn=collate_fn_nl,
                pin_memory=True)

        self.test_data_loader = DataLoader(
                test_data_engine,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True,
                collate_fn=collate_fn_nl,
                pin_memory=True)

    def get_log_prob(self, sentences):
        """
        args:
            sentences: tensor, shape [batch_size, seq_length]
                       sentences without adding _BOS
                       or shape [batch_size, seq_length, vocab_size]
                       one-hot vectors

        returns:
            log_probs: tensor, shape [batch_size]
                       log-probability of sentences
        """
        self.lm.eval()
        sentences = sentences.detach().clone()
        if len(sentences.size()) == 3:
            _, sentences = torch.max(sentences, dim=-1)
        bos = torch.full_like(sentences[:, 0], _BOS, dtype=torch.long).unsqueeze(1)
        inputs = torch.cat((bos, sentences[:, :-1]), dim=1).to(self.device)
        targets = sentences.clone().detach()

        with torch.no_grad():
            logits = self.lm(inputs)

        return self.get_log_prob_logits(logits, targets)

    def get_log_prob_logits(self, logits, labels):
        """
        args:
            logits: tensor, shape [batch_size, seq_length, vocab_size]
            labels: tensor, shape [batch_size, seq_length]

        returns:
            log_probs: tensor, shape [batch_size]
                       log-probability of sentences
        """
        log_probs = F.log_softmax(logits.cpu().detach(), dim=-1)
        # make log_probs for _PAD and _EOS be 0 so they won't be counted
        log_probs[:, :, _PAD] = 0
        log_probs[:, :, _EOS] = 0
        log_probs = torch.gather(log_probs, dim=-1, index=labels.cpu().long().unsqueeze(-1))
        log_probs = log_probs.squeeze(-1).sum(dim=1)
        return log_probs

    def train(self, epochs, batch_size, criterion, save_epochs=10):
        for idx in range(1, epochs+1):
            epoch_loss = 0
            batch_amount = 0

            pbar = tqdm(self.train_data_loader, desc="Iteration", ascii=True, dynamic_ncols=True)
            for b_idx, batch in enumerate(pbar):
                batch_loss, batch_logits = self.run_batch(
                        batch, criterion, testing=False)
                epoch_loss += batch_loss
                batch_amount += 1
                pbar.set_postfix(Loss="{:.5f}".format(epoch_loss / batch_amount))

            epoch_loss /= batch_amount
            print_time_info("Epoch {} finished, training loss {}".format(
                    idx, epoch_loss))

            self.test(batch_size, criterion)
            if idx % save_epochs == 0:
                print_time_info("Epoch {}: save model...".format(idx))
                self.save_model(self.model_dir)

    def test(self, batch_size, criterion, result_path=None):
        if result_path and os.path.exists(result_path):
            os.remove(result_path)

        with torch.no_grad():
            test_loss = 0
            batch_amount = 0
            for b_idx, batch in enumerate(tqdm(self.test_data_loader)):
                batch_loss, batch_logits = self.run_batch(
                        batch, criterion, testing=True,
                        result_path=result_path)
                test_loss += batch_loss
                batch_amount += 1

            test_loss /= batch_amount
            print_time_info("testing finished, testing loss {}".format(test_loss))

    def run_batch(self, batch, criterion, testing=False, result_path=None):
        if testing:
            self.lm.eval()
        else:
            self.lm.train()

        encoder_input, decoder_label, refs, sf_data = batch

        inputs = torch.from_numpy(encoder_input).to(self.device)
        targets = torch.from_numpy(decoder_label).long()

        logits = self.lm(inputs)
        batch_size, seq_length, vocab_size = logits.size()
        loss = criterion(
            logits.cpu().contiguous().view(-1, vocab_size),
            targets.contiguous().view(-1)
        )

        if not testing:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if testing and result_path:
            self._record_test_result(
                result_path, encoder_input,
                decoder_label, sf_data, logits, targets
            )

        return loss, logits

    def save_model(self, model_dir):
        path = os.path.join(model_dir, "lm.ckpt")
        torch.save(self.lm, path)
        print_time_info("Save model successfully")

    def load_model(self, model_dir):
        path = os.path.join(model_dir, "lm.ckpt")
        if not os.path.exists(path):
            print_time_info("Loading failed, start training from scratch...")
        else:
            self.lm = torch.load(path, map_location=self.device)
            print_time_info("Load model from {} successfully".format(model_dir))

    @classmethod
    def load_pretrained(cls,
                        model_dir,
                        train_data_engine,
                        test_data_engine,
                        device=None):
        config_path = os.path.join(model_dir, "lm_config")
        args = dict()
        for line in open(config_path):
            name, value = line.strip().split(': ', maxsplit=1)
            args[name] = value

        lm = cls(
                batch_size=int(args['batch_size']),
                optimizer=args['optimizer'],
                learning_rate=float(args['learning_rate']),
                train_data_engine=train_data_engine,
                test_data_engine=test_data_engine,
                dim_hidden=int(args['hidden_size']),
                dim_embedding=int(args['embedding_dim']),
                vocab_size=int(args['vocab_size']) + 4,
                n_layers=int(args['n_layers']),
                model_dir=args['model_dir'],
                log_dir=args['log_dir'],
                is_load=True,
                replace_model=int(args['replace_model']),
                device=device,
                dir_name=args['dir_name']
        )
        return lm

    def _record_test_result(self, result_path, encoder_input,
                            decoder_label, sf_data, logits, targets):
        def untokenize(sentences, sf_data):
            return [
                self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
                for idx, sent in enumerate(sentences)
            ]

        no_sf_data = [
            {"name": "NAMETOKEN", "near": "NEARTOKEN"}
            for _ in range(len(encoder_input))
        ]

        encoder_input_original = untokenize(encoder_input, no_sf_data)
        encoder_input = untokenize(encoder_input, sf_data)
        decoder_label_original = untokenize(decoder_label, no_sf_data)
        decoder_label = untokenize(decoder_label, sf_data)

        _, lm_output = torch.max(logits.cpu().detach(), dim=-1)
        lm_output_original = untokenize(lm_output, no_sf_data)
        lm_output = untokenize(lm_output, sf_data)
        lm_log_prob = list(self.get_log_prob_logits(
                logits.cpu().detach(), targets).numpy())

        with open(result_path, 'a') as file:
            for idx in range(len(encoder_input)):
                file.write("---------\n")
                file.write(f"Data {idx}\n")
                file.write(f"encoder input: {' '.join(encoder_input[idx])}\n")
                file.write(f"decoder label: {' '.join(decoder_label[idx])}\n")
                file.write(f"lm output: {' '.join(lm_output[idx])}\n")
                file.write(f"encoder input original: {' '.join(encoder_input_original[idx])}\n")
                file.write(f"decoder label original: {' '.join(decoder_label_original[idx])}\n")
                file.write(f"lm output original: {' '.join(lm_output_original[idx])}\n")
                file.write(f"lm log-prob: {lm_log_prob[idx]}\n")
