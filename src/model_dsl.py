import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import time
import random
import numpy as np
import os
import math

from module import NLURNN, NLGRNN, LMRNN
from utils import *
from text_token import _UNK, _PAD, _BOS, _EOS
from model_utils import collate_fn_nlg, collate_fn_nlu, build_optimizer, get_device
from logger import Logger

from tqdm import tqdm


class DSL:
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
            attr_vocab_size=None,
            n_layers=1,
            bidirectional=False,
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
        self.attr_vocab_size = attr_vocab_size
        self.dir_name = dir_name

        self.device = get_device(device)

        self.nlu = NLURNN(
                dim_embedding=dim_embedding,
                dim_hidden=dim_hidden,
                attr_vocab_size=attr_vocab_size,
                vocab_size=vocab_size,
                n_layers=n_layers,
                bidirectional=bidirectional)

        self.nlg = NLGRNN(
                dim_embedding=dim_embedding,
                dim_hidden=dim_hidden,
                attr_vocab_size=attr_vocab_size,
                vocab_size=vocab_size,
                n_layers=n_layers,
                bidirectional=False)

        self.nlu.to(self.device)
        self.nlg.to(self.device)

        # Initialize data loaders and optimizers
        self.train_data_engine = train_data_engine
        self.test_data_engine = test_data_engine
        self.train_data_loader = DataLoader(
                train_data_engine,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=True,
                collate_fn=collate_fn_nlg,
                pin_memory=True)

        self.test_data_loader = DataLoader(
                test_data_engine,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True,
                collate_fn=collate_fn_nlg,
                pin_memory=True)

        # nlu parameters optimization
        self.nlu_parameters = filter(
                lambda p: p.requires_grad, self.nlu.parameters())
        self.nlu_optimizer = build_optimizer(
                optimizer, self.nlu_parameters,
                learning_rate)
        # nlg parameters optimization
        self.nlg_parameters = filter(
                lambda p: p.requires_grad, self.nlg.parameters())
        self.nlg_optimizer = build_optimizer(
                optimizer, self.nlg_parameters,
                learning_rate)

        print_time_info("Model create complete")

        self.model_dir, self.log_dir = handle_model_dirs(
            model_dir, log_dir, dir_name, replace_model, is_load
        )

        if is_load:
            self.load_model(self.model_dir)

        self.train_log_path = os.path.join(self.log_dir, "train_log.csv")
        self.valid_log_path = os.path.join(
                self.log_dir, "valid_log.csv")

        with open(self.train_log_path, 'w') as file:
            file.write("epoch,nlu_loss,nlg_loss,micro_f1,"
                       "bleu,rouge(1,2,L,BE)\n")
        with open(self.valid_log_path, 'w') as file:
            file.write("epoch,nlu_loss,nlg_loss,micro_f1, "
                       "bleu,rouge(1,2,L,BE)\n")

        # Initialize batch count
        self.batches = 0

    def train(self, epochs, batch_size, criterion,
              save_epochs=10,
              teacher_forcing_ratio=0.5,
              tf_decay_rate=0.9,
              max_norm=0.25):

        self.batches = 0

        for idx in range(1, epochs+1):
            epoch_nlg_loss = epoch_nlu_loss = epoch_dual_loss = 0
            batch_amount = 0
            nlu_scorer = MultilabelScorer(f1_per_sample=False)
            nlg_scorer = SequenceScorer()

            pbar = tqdm(
                self.train_data_loader,
                total=len(self.train_data_loader),
                dynamic_ncols=True
            )

            for batch in pbar:
                self.batches += 1
                nlg_logits, nlg_outputs, nlg_targets = self.run_nlg_batch(
                        batch,
                        scorer=nlg_scorer,
                        testing=False,
                        teacher_forcing_ratio=teacher_forcing_ratio
                )

                nlu_logits, nlu_targets = self.run_nlu_batch(
                        batch,
                        scorer=nlu_scorer,
                        testing=False
                )

                nlg_loss, nlu_loss, dual_loss = criterion(
                        nlg_logits.cpu(),
                        nlg_outputs.cpu(),
                        nlu_logits.cpu(),
                        nlg_targets.cpu(),
                        nlu_targets.cpu()
                )

                nlg_loss.backward(retain_graph=True)
                nlu_loss.backward(retain_graph=False)
                self.nlu_optimizer.step()
                self.nlg_optimizer.step()
                self.nlu_optimizer.zero_grad()
                self.nlg_optimizer.zero_grad()

                batch_amount += 1
                epoch_nlu_loss += nlu_loss.item()
                epoch_nlg_loss += nlg_loss.item()
                epoch_dual_loss += dual_loss.item()
                pbar.set_postfix(
                        ULoss="{:.4f}".format(epoch_nlu_loss / batch_amount),
                        GLoss="{:.3f}".format(epoch_nlg_loss / batch_amount),
                        DLoss="{:.4f}".format(epoch_dual_loss / batch_amount),
                )

            nlg_scorer.print_avg_scores()
            nlu_scorer.print_avg_scores()

            # save model
            if idx % save_epochs == 0:
                print_time_info(f"Epoch {idx}: save model...")
                self.save_model(self.model_dir)

            self._record_log(
                epoch=idx,
                testing=False,
                nlu_loss=epoch_nlu_loss,
                nlg_loss=epoch_nlg_loss,
                nlu_scorer=nlu_scorer,
                nlg_scorer=nlg_scorer
            )

            self.test(
                batch_size=batch_size,
                criterion=criterion,
                epoch=idx
            )

            teacher_forcing_ratio *= tf_decay_rate
            criterion.epoch_end()

        return (
            nlg_loss / batch_amount,
            nlu_loss / batch_amount,
            nlg_scorer,
            nlu_scorer,
        )

    def test(self, batch_size,
             criterion, epoch=-1):

        batch_amount = 0

        nlu_scorer = MultilabelScorer(f1_per_sample=False)
        nlu_loss = 0
        nlg_scorer = SequenceScorer()
        nlg_loss = 0
        dual_loss = 0
        for b_idx, batch in enumerate(tqdm(self.test_data_loader)):
            with torch.no_grad():
                nlu_logits, nlu_targets = self.run_nlu_batch(
                        batch,
                        scorer=nlu_scorer,
                        testing=True
                )
                nlg_logits, nlg_outputs, nlg_targets = self.run_nlg_batch(
                        batch,
                        scorer=nlg_scorer,
                        testing=True,
                        teacher_forcing_ratio=0.0,
                        result_path=os.path.join(
                            os.path.join(self.log_dir, "validation"),
                            "test.txt"
                        )
                )
                batch_nlg_loss, batch_nlu_loss, batch_dual_loss = criterion(
                        nlg_logits.cpu(),
                        nlg_outputs.cpu(),
                        nlu_logits.cpu(),
                        nlg_targets.cpu(),
                        nlu_targets.cpu()
                )

            nlu_loss += batch_nlu_loss.item()
            nlg_loss += batch_nlg_loss.item()
            dual_loss += batch_dual_loss.item()
            batch_amount += 1

        nlu_loss /= batch_amount
        nlu_scorer.print_avg_scores()
        nlg_loss /= batch_amount
        nlg_scorer.print_avg_scores()
        dual_loss /= batch_amount

        self._record_log(
            epoch=epoch,
            testing=True,
            nlu_loss=nlu_loss,
            nlg_loss=nlg_loss,
            dual_loss=dual_loss,
            nlu_scorer=nlu_scorer,
            nlg_scorer=nlg_scorer
        )

        with open("test_results.txt", 'a') as file:
            file.write("{}\n".format(self.dir_name))
            nlg_scorer.write_avg_scores_to_file(file)
            nlu_scorer.write_avg_scores_to_file(file)

    def run_nlu_batch(self, batch, scorer=None,
                      testing=False, result_path=None):
        if testing:
            self.nlu.eval()
        else:
            self.nlu.train()

        decoder_label, encoder_input, refs, sf_data = batch

        inputs = torch.from_numpy(encoder_input).to(self.device)
        targets = self._sequences_to_nhot(decoder_label, self.attr_vocab_size)
        targets = torch.from_numpy(targets).float()

        logits = self.nlu(inputs)
        prediction = (torch.sigmoid(logits.detach().cpu()) >= 0.5)
        prediction = prediction.clone().numpy()
        if scorer:
            targets_clone = targets.detach().cpu().long().numpy()
            scorer.update(targets_clone, prediction)

        if testing and result_path:
            self._record_nlu_test_result(
                result_path,
                encoder_input,
                decoder_label,
                prediction
            )

        return logits, targets

    def run_nlg_batch(self, batch, scorer=None,
                      testing=False, teacher_forcing_ratio=0.5,
                      result_path=None):
        if testing:
            self.nlg.eval()
        else:
            self.nlg.train()

        encoder_input, decoder_label, refs, sf_data = batch

        attrs = self._sequences_to_nhot(encoder_input, self.attr_vocab_size)
        attrs = torch.from_numpy(attrs).to(self.device)
        labels = torch.from_numpy(decoder_label).to(self.device)

        # logits.size() == (batch_size, 1, seq_length, vocab_size)
        # outputs.size() == (batch_size, 1, seq_length, vocab_size) one-hot vectors
        # Note that outputs are still in computational graph
        logits, outputs = self.nlg(
            attrs, _BOS, labels, beam_size=1,
            tf_ratio=teacher_forcing_ratio if not testing else 0.0
        )

        logits = logits.squeeze(1)
        outputs = outputs.squeeze(1)
        batch_size, seq_length, vocab_size = logits.size()

        outputs_indices = outputs.detach().cpu().clone().numpy()
        outputs_indices = np.argmax(outputs_indices, axis=-1)
        if scorer:
            labels_clone = labels.detach().cpu().numpy()
            scorer.update(labels_clone, refs, outputs_indices)

        if testing and result_path:
            self._record_nlg_test_result(
                result_path,
                encoder_input,
                decoder_label,
                sf_data,
                outputs_indices
            )
        return logits, outputs, labels

    def save_model(self, model_dir):
        nlu_path = os.path.join(model_dir, "nlu.ckpt")
        nlg_path = os.path.join(model_dir, "nlg.ckpt")
        torch.save(self.nlu, nlu_path)
        torch.save(self.nlg, nlg_path)
        print_time_info("Save model successfully")

    def load_model(self, model_dir):
        # Get the latest modified model (files or directory)
        nlu_path = os.path.join(model_dir, "nlu.ckpt")
        nlg_path = os.path.join(model_dir, "nlg.ckpt")

        if not os.path.exists(nlu_path) or not os.path.exists(nlg_path):
            print_time_info("Loading failed, start training from scratch...")
        else:
            self.nlu = torch.load(nlu_path, map_location=self.device)
            self.nlg = torch.load(nlg_path, map_location=self.device)
            print_time_info(f"Load model from {model_dir} successfully")

    def _sequences_to_nhot(self, seqs, vocab_size):
        """
        args:
            seqs: list of list of word_ids
            vocab_size: int

        outputs:
            labels: np.array of shape [batch_size, vocab_size]
        """
        labels = np.zeros((len(seqs), vocab_size), dtype=np.int)
        for bid, seq in enumerate(seqs):
            for word in seq:
                labels[bid][word] = 1
        return labels

    def _record_log(self,
                    epoch,
                    testing,
                    nlu_loss=None,
                    nlg_loss=None,
                    dual_loss=None,
                    nlu_scorer=None,
                    nlg_scorer=None):
        filename = self.valid_log_path if testing else self.train_log_path
        nlu_loss = 'None' if nlu_loss is None else '{:.4f}'.format(nlu_loss)
        nlg_loss = 'None' if nlg_loss is None else '{:.3f}'.format(nlg_loss)
        dual_loss = 'None' if dual_loss is None else '{:.4f}'.format(dual_loss)
        if nlu_scorer is not None:
            micro_f1, _ = nlu_scorer.get_avg_scores()
            micro_f1 = '{:.4f}'.format(micro_f1)
        else:
            micro_f1 = '-1.0'
        if nlg_scorer is not None:
            _, bleu, _, rouge, _ = nlg_scorer.get_avg_scores()
            bleu = '{:.4f}'.format(bleu)
            rouge = ' '.join(['{:.4f}'.format(s) for s in rouge])
        else:
            bleu, rouge = '-1.0', '-1.0 -1.0 -1.0'
        with open(filename, 'a') as file:
            file.write(f"{epoch},{nlu_loss},{nlg_loss},{micro_f1},"
                       f"{bleu},{rouge},{dual_loss}\n")

    def _record_nlu_test_result(self,
                                result_path,
                                encoder_input,
                                decoder_label,
                                prediction):
        '''
        encoder_input = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx], is_token=True)
            for idx, sent in enumerate(encoder_input)
        ]
        decoder_label = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            for idx, sent in enumerate(decoder_label)
        ]
        decoder_result = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            for idx, sent in enumerate(decoder_result)
        ]

        with open(result_path, 'a') as file:
            for idx in range(len(encoder_input)):
                file.write("---------\n")
                file.write(f"Data {idx}\n")
                file.write(f"encoder input: {' '.join(encoder_input[idx])}\n")
                file.write(f"decoder output: {' '.join(decoder_result[idx])}\n")
                file.write(f"decoder label: {' '.join(decoder_label[idx])}\n")
        '''
        pass

    def _record_nlg_test_result(self, result_path, encoder_input,
                            decoder_label, sf_data, decoder_result):
        encoder_input = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx], is_token=True)
            for idx, sent in enumerate(encoder_input)
        ]
        decoder_label = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            for idx, sent in enumerate(decoder_label)
        ]
        decoder_result = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            for idx, sent in enumerate(decoder_result)
        ]

        with open(result_path, 'a') as file:
            for idx in range(len(encoder_input)):
                file.write("---------\n")
                file.write(f"Data {idx}\n")
                file.write(f"encoder input: {' '.join(encoder_input[idx])}\n")
                file.write(f"decoder output: {' '.join(decoder_result[idx])}\n")
                file.write(f"decoder label: {' '.join(decoder_label[idx])}\n")
