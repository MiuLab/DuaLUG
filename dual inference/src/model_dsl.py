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

from module_dim import DIMLoss, NLURNN, NLGRNN, LMRNN
from utils import *
from text_token import _UNK, _PAD, _BOS, _EOS
from model_utils import collate_fn_nlg, collate_fn_nlu, build_optimizer, get_device
from logger import Logger

from tqdm import tqdm

def _has_inf_or_nan(x):
    try:
        # if x is half, the .float() incurs an additional deep copy, but it's necessary if 
        # Pytorch's .sum() creates a one-element tensor of the same type as x 
        # (which is true for some recent version of pytorch).
        cpu_sum = float(x.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # cpu_sum = float(x.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True
        return False


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
            dir_name='test',
            with_intent=True
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
        self.with_intent = with_intent
        self.device = get_device(device)

        self.nlu = NLURNN(
                dim_embedding=dim_embedding,
                dim_hidden=dim_hidden,
                vocab_size=train_data_engine.tokenizer.get_vocab_size(),
                slot_vocab_size=len(train_data_engine.nlu_slot_vocab),
                intent_vocab_size=len(train_data_engine.intent_vocab),
                n_layers=n_layers,
                bidirectional=bidirectional)

        self.nlg = NLGRNN(
                dim_embedding=dim_embedding,
                dim_hidden=dim_hidden,
                vocab_size=train_data_engine.tokenizer.get_vocab_size(),
                n_slot_key=len(train_data_engine.nlg_slot_vocab),
                n_intent=len(train_data_engine.intent_vocab),
                n_layers=n_layers,
                bidirectional=False,
                batch_size=batch_size)

        self.nlu.to(self.device)
        self.nlg.to(self.device)

        # Initialize data loaders and optimizers
        self.train_data_engine = train_data_engine
        self.test_data_engine = test_data_engine
        self.test_result_path = os.path.join(self.log_dir, "test_result.txt")
        
        """
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
        """
        self.train_nlu_data_loader = DataLoader(
                train_data_engine,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=True,
                collate_fn=train_data_engine.collate_fn_nlu,
                pin_memory=True)
        self.train_nlg_data_loader = DataLoader(
                train_data_engine,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=True,
                collate_fn=train_data_engine.collate_fn_nlg,
                pin_memory=True)

        self.test_nlu_data_loader = DataLoader(
                test_data_engine,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True,
                collate_fn=test_data_engine.collate_fn_nlu,
                pin_memory=True)

        self.test_nlg_data_loader = DataLoader(
                test_data_engine,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True,
                collate_fn=test_data_engine.collate_fn_nlg,
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

        print_time_info("Model create completed.")

        self.train_log_path = os.path.join(self.log_dir, "train_log.csv")
        self.valid_log_path = os.path.join(self.log_dir, "valid_log.csv")

        with open(self.train_log_path, 'w') as file:
            file.write("epoch,nlu_loss,nlg_loss,intent_acc,slot_f1,bleu,rouge(1,2,L)\n")
        with open(self.valid_log_path, 'w') as file:
            file.write("epoch,nlu_loss,nlg_loss,intent_acc,slot_f1,bleu,rouge(1,2,L)\n")

        # Initialize batch count
        self.batches = 0

    def train(self, epochs, batch_size, criterion, criterion_nlu, criterion_nlg,
              save_epochs=10,
              teacher_forcing_ratio=0.5,
              tf_decay_rate=0.9,
              max_norm=0.25):

        self.batches = 0

        for idx in range(1, epochs+1):
            epoch_nlg_loss = epoch_nlu_loss = epoch_dual_loss = 0
            batch_amount = 0
            nlu_scorer = IntentPredSlotFillScorer()
            # nlu_scorer = MultilabelScorer(f1_per_sample=False)
            nlg_scorer = SequenceScorer()

            """
            pbar = tqdm(
                self.train_data_loader,
                total=len(self.train_data_loader),
                dynamic_ncols=True
            )
            """
            pbar = tqdm(
                zip(self.train_nlg_data_loader, self.train_nlu_data_loader),
                total=len(self.train_nlg_data_loader),
                dynamic_ncols=True
            )

            # for batch in pbar:
            for batch_nlg, batch_nlu in pbar:
                self.batches += 1
                nlg_logits, nlg_outputs, nlg_targets = self.run_nlg_batch(
                        batch_nlg,
                        scorer=nlg_scorer,
                        testing=False,
                        teacher_forcing_ratio=teacher_forcing_ratio
                )

                nlu_logits, slot_prediction, nlu_targets, intent_logits, intent_prediction, intent_targets = self.run_nlu_batch(
                        batch_nlu,
                        scorer=nlu_scorer,
                        testing=False
                )
                
                # TODO: change criterion input, nlu: intent+slot-value pairs
                # slot value pairs 
                # semantic_frames = list()
                # for slot_seq, word_seq in zip(slot_prediction, batch_nlu['inputs'].clone().numpy()):
                #     semantic_frames.append(self.train_data_engine.fuzzy_mapping_slots_to_semantic_frame(slot_seq, word_seq))
                # pseudo_nlg_inputs = self.train_data_engine.batch_semantic_frame_to_nlg_input(semantic_frames)
                # TODO: remember to add intent
                # intent_prediction

                attrs = (
                    batch_nlg['slot_key'].cuda(),
                    batch_nlg['slot_key_lens'],
                    batch_nlg['slot_value'].cuda(),
                    batch_nlg['slot_value_lens'],
                    batch_nlg['intent'].cuda()
                )
                # pseudo_attrs = (
                #     pseudo_nlg_inputs['slot_key'].clone().detach().to(self.device),
                #     torch.tensor(pseudo_nlg_inputs['slot_key_lens']).to(self.device),
                #     pseudo_nlg_inputs['slot_value'].clone().detach().to(self.device),
                #     torch.tensor(pseudo_nlg_inputs['slot_value_lens']).to(self.device),
                #     torch.tensor(intent_prediction).to(self.device))

                # pdb.set_trace()
                nlg_loss, nlu_loss, dual_loss = criterion(
                        nlg_logits.cpu(),
                        nlg_outputs.cpu(),
                        nlg_targets.cpu(),
                        nlu_logits.cpu(),
                        nlu_targets.cpu(),
                        intent_logits.cpu(),
                        intent_targets.cpu(),
                        attrs
                )
                # pdb.set_trace()

                if _has_inf_or_nan(nlg_loss) or _has_inf_or_nan(nlu_loss):
                    print("Overflow! Skip this batch %d."%self.batches)
                else:
                    nlg_loss.backward(retain_graph=True)
                    nlu_loss.backward(retain_graph=False)
                    clip_grad_norm_(self.nlg_parameters, 1.0)
                    clip_grad_norm_(self.nlu_parameters, 1.0)
                    self.nlg_optimizer.step()
                    self.nlu_optimizer.step()
                    epoch_nlu_loss += nlu_loss.item()
                    epoch_nlg_loss += nlg_loss.item()
                    epoch_dual_loss += dual_loss.item()
                    batch_amount += 1
                self.nlg_optimizer.zero_grad()
                self.nlu_optimizer.zero_grad()
                

                pbar.set_postfix(
                        ULoss="{:.4f}".format((epoch_nlu_loss / batch_amount) if batch_amount>0 else 0.0),
                        GLoss="{:.3f}".format((epoch_nlg_loss / batch_amount) if batch_amount>0 else 0.0),
                        DLoss="{:.4f}".format((epoch_dual_loss / batch_amount) if batch_amount>0 else 0.0),
                )

            nlg_scorer.print_avg_scores()
            nlu_scorer.print_avg_scores()

            # save model
            if idx % save_epochs == 0:
                print_time_info("Epoch {}: save model...".format(idx))
                self.save_model(self.model_dir)

            self._record_log(
                epoch=idx,
                testing=False,
                nlu_loss=nlu_loss,
                nlg_loss=nlg_loss,
                nlu_scorer=nlu_scorer,
                nlg_scorer=nlg_scorer
            )

            self.test(
                batch_size=batch_size,
                criterion_nlg=criterion_nlg,
                criterion_nlu=criterion_nlu,
                test_nlu=True,
                test_nlg=True,
                epoch=idx
            )

            criterion_nlu.epoch_end()
            criterion_nlg.epoch_end()

            teacher_forcing_ratio *= tf_decay_rate

    def test(self, batch_size,
             criterion_nlu, criterion_nlg,
             test_nlu=True, test_nlg=True,
             sample_size=1, epoch=-1):

        nlu_loss = nlg_loss = None
        nlu_scorer = nlg_scorer = None

        batch_amount = 0

        if test_nlu:
            # nlu_scorer = MultilabelScorer()
            nlu_scorer = IntentPredSlotFillScorer(intent_acc=self.with_intent)
            nlu_loss = 0
            for b_idx, batch in enumerate(tqdm(self.test_nlu_data_loader)):
                with torch.no_grad():
                    batch_loss, batch_logits, slot_prediction, intent_logits, intent_prediction, _, _, _ = self.run_test_nlu_batch(
                            batch,
                            criterion_nlu,
                            scorer=nlu_scorer,
                            testing=True
                    )
                nlu_loss += batch_loss.item()
                batch_amount += 1

            nlu_loss /= batch_amount
            nlu_scorer.print_avg_scores()

        batch_amount = 0

        if test_nlg:
            nlg_scorer = SequenceScorer()
            nlg_loss = 0
            for b_idx, batch in enumerate(tqdm(self.test_nlg_data_loader)):
                with torch.no_grad():
                    batch_loss, batch_logits, batch_decode_result, _, _, _, _ = self.run_test_nlg_batch(
                            batch,
                            criterion_nlg,
                            scorer=nlg_scorer,
                            testing=True,
                            teacher_forcing_ratio=0.0,
                            beam_size=sample_size,
                            result_path=os.path.join(
                                os.path.join(self.log_dir, "validation"),
                                "test.txt"
                            )
                    )

                nlg_loss += batch_loss.item()
                batch_amount += 1

            nlg_loss /= batch_amount
            nlg_scorer.print_avg_scores()

        self._record_log(
            epoch=epoch,
            testing=True,
            nlu_loss=nlu_loss,
            nlg_loss=nlg_loss,
            nlu_scorer=nlu_scorer,
            nlg_scorer=nlg_scorer
        )

        # with open("test_results.txt", 'a') as file:
        #     if test_nlu or test_nlg:
        #         file.write("{}\n".format(self.dir_name))
        #     if test_nlg:
        #         nlg_scorer.write_avg_scores_to_file(file)
        #     if test_nlu:
        #         nlu_scorer.write_avg_scores_to_file(file)
        # if test_nlu or test_nlg:
        #     file.write("{}\n".format(self.dir_name))
        if test_nlg:
            nlg_scorer.write_avg_scores_to_file(self.test_result_path)
        if test_nlu:
            nlu_scorer.write_avg_scores_to_file(self.test_result_path)

    # def run_nlu_batch(self, batch, scorer=None, testing=False, result_path=None):
    def run_nlu_batch(self, batch, scorer=None,
                      testing=False, optimize=True, max_norm=None, beam_size=1,
                      teacher_forcing_ratio=0.5, retain_graph=False, result_path=None, sample_size=1,
                      supervised=True):
        if testing:
            self.nlu.eval()
        else:
            self.nlu.train()

        inputs = batch['inputs'].to(self.device)
        targets = batch['labels'].clone().detach().to(self.device)
        intent_targets = batch['intent'].clone()

        slot_logits, outputs, decisions, intent_logits = self.nlu(
            inputs,
            _BOS,
            labels=targets, 
            beam_size=beam_size,
            tf_ratio=teacher_forcing_ratio if not testing else 0.0
        )

        outputs_indices = decisions[:, 0].detach().cpu().clone().numpy()
        outputs_indices = np.argmax(outputs_indices, axis=-1)
        slot_prediction = outputs_indices

        if self.with_intent:
            intent_prediction = torch.argmax(intent_logits.detach().cpu(), dim=1)
            intent_prediction = intent_prediction.clone().numpy()


        if scorer:
            # slot filling
            targets_clone = targets.detach().cpu().long().numpy()
            targets_clone = [
                self.train_data_engine.untokenize_nlu_slot_seq(target) 
                for target in targets_clone
            ]
            slot_prediction = [
                self.train_data_engine.untokenize_nlu_slot_seq(prediction)
                for prediction in slot_prediction
            ]
            scorer.update(
                targets_clone,
                slot_prediction,
                intent_labels=batch['intent'].clone().cpu().numpy() if self.with_intent else None,
                intent_prediction=intent_prediction if self.with_intent else None
            )

        # return logits, targets
        # if sample_size > 1:
        #     samples = self._sample_nlu_output(logits, sample_size)
        # else:
            # samples = self._st_sigmoid(logits, hard=True).unsqueeze(1)
        #     samples = self._st_sigmoid(slot_logits, hard=True, argmax=True).unsqueeze(1)

        # loss for slot filling and intent prediction
        """
        sup_loss, rl_loss, nlu_joint_prob, reward = criterion(
            slot_logits.cpu().unsqueeze(1),#.expand(-1, sample_size, -1),
            batch['labels'].clone().detach().cpu(),
            decisions=samples.cpu(),
            intent_logits=intent_logits.cpu().unsqueeze(1) if self.with_intent else None,#.expand(-1, sample_size, -1),
            intent_targets=batch['intent'].cpu() if self.with_intent else None,
            n_supervise=1,
            calculate_reward=reinforce,
        )
        """
        # return sup_loss + rl_loss, slot_logits, intent_logits, samples, nlu_joint_prob, reward
        return slot_logits, outputs_indices, targets, intent_logits, intent_prediction, intent_targets



    def run_nlg_batch(self, batch, scorer=None,
                      testing=False, optimize=True, max_norm=None, beam_size=1,
                      teacher_forcing_ratio=0.5, retain_graph=False, result_path=None, sample_size=1,
                      supervised=True, reinforce=False):
        if testing:
            self.nlg.eval()
        else:
            self.nlg.train()

        # encoder_input, decoder_label, refs, sf_data = batch

        # attrs = self._sequences_to_nhot(encoder_input, self.attr_vocab_size)
        # attrs = torch.from_numpy(attrs).to(self.device)
        # labels = torch.from_numpy(decoder_label).to(self.device)
        attrs = (
            batch['slot_key'].clone().detach().to(self.device),
            torch.tensor(batch['slot_key_lens']).to(self.device),
            batch['slot_value'].clone().detach().to(self.device),
            torch.tensor(batch['slot_value_lens']).to(self.device),
            batch['intent'].clone().detach().to(self.device))
        labels = batch['target'].clone().detach().to(self.device)
        refs = batch['multi_refs']


        # logits.size() == (batch_size, 1, seq_length, vocab_size)
        # outputs.size() == (batch_size, 1, seq_length, vocab_size) one-hot vectors
        # Note that outputs are still in computational graph

        logits, outputs, decisions, semantic_embs = self.nlg(
            attrs,
            _BOS,
            labels, 
            beam_size=beam_size,
            tf_ratio=teacher_forcing_ratio if not testing else 0.0,
            # st=nlg_st
        )


        batch_size, _, seq_length, vocab_size = logits.size()

        outputs_indices = decisions[:, 0].detach().cpu().clone().numpy()
        outputs_indices = np.argmax(outputs_indices, axis=-1)
        if scorer:
            labels_clone = labels.detach().cpu().numpy()
            scorer.update(labels_clone, refs, outputs_indices)

        # pdb.set_trace()
        """
        sup_loss, rl_loss, nlg_joint_prob, reward = criterion(
            logits.cpu(),
            labels.cpu(),
            decisions=decisions.cpu(),
            n_supervise=1,
            calculate_reward=reinforce,
            # inputs=attrs
        )
        has_rl = isinstance(rl_loss, torch.Tensor)
        if not testing:
            if supervised and isinstance(sup_loss, torch.Tensor):
                sup_loss.backward(retain_graph=(retain_graph or has_rl or self.dim_loss))
            if reinforce and has_rl:
                rl_loss.backward(retain_graph=retain_graph)
            if optimize and not self.dim_loss:
                if max_norm:
                    clip_grad_norm_(self.nlg_parameters, max_norm)
                self.nlg_optimizer.step()
                self.nlg_optimizer.zero_grad()
                # self.nlu_optimizer.zero_grad()
        """
        """
        if testing and result_path:
            self._record_nlg_test_result(
                result_path,
                encoder_input,
                decoder_label,
                sf_data,
                outputs_indices
            )
        """
        return logits, outputs, labels

    def run_test_nlu_batch(self, batch, criterion, scorer=None,
                      testing=False, optimize=True, max_norm=None, beam_size=1,
                      teacher_forcing_ratio=0.5, retain_graph=False, result_path=None, sample_size=1,
                      supervised=True, reinforce=False):
        if testing:
            self.nlu.eval()
        else:
            self.nlu.train()

        # return {'inputs': inputs, 'labels': labels, 'intent': intent}
        # encoder_input, decoder_label, refs, sf_data = batch
        # inputs = torch.from_numpy(encoder_input).to(self.device)
        inputs = batch['inputs'].to(self.device)
        # targets = self._sequences_to_nhot(decoder_label, self.attr_vocab_size)
        # targets = torch.from_numpy(targets).float()
        targets = batch['labels'].clone().detach().to(self.device)

        # pdb.set_trace()
        # logits = self.nlu(inputs)
        slot_logits, outputs, decisions, intent_logits = self.nlu(
            inputs,
            _BOS,
            labels=targets, 
            beam_size=beam_size,
            tf_ratio=teacher_forcing_ratio if not testing else 0.0
        )
        # pdb.set_trace()
        # prediction = (torch.sigmoid(logits.detach().cpu()) >= 0.5)
        # prediction = prediction.clone().numpy()
        # slot_prediction = torch.argmax(slot_logits.detach().cpu(), dim=2)
        # slot_prediction = slot_prediction.clone().numpy()
        
        outputs_indices = decisions[:, 0].detach().cpu().clone().numpy()
        outputs_indices = np.argmax(outputs_indices, axis=-1)
        slot_prediction = outputs_indices

        if self.with_intent:
            intent_prediction = torch.argmax(intent_logits.detach().cpu(), dim=1)
            intent_prediction = intent_prediction.clone().numpy()


        if scorer:
            # slot filling
            targets_clone = targets.detach().cpu().long().numpy()
            targets_clone = [
                self.train_data_engine.untokenize_nlu_slot_seq(target) 
                for target in targets_clone
            ]
            slot_prediction = [
                self.train_data_engine.untokenize_nlu_slot_seq(prediction)
                for prediction in slot_prediction
            ]
            scorer.update(
                targets_clone,
                slot_prediction,
                intent_labels=batch['intent'].clone().cpu().numpy() if self.with_intent else None,
                intent_prediction=intent_prediction if self.with_intent else None
            )

        if sample_size > 1:
            samples = self._sample_nlu_output(logits, sample_size)
        else:
            # samples = self._st_sigmoid(logits, hard=True).unsqueeze(1)
            samples = self._st_sigmoid(slot_logits, hard=True, argmax=True).unsqueeze(1)

        # torch.zeros_like(slot_logits).scatter_(2, targets.unsqueeze(2), 1)
        # torch.zeros_like(slot_logits).scatter_(1,targets.unsqueeze(2),1)
        # pdb.set_trace()
        # loss for slot filling and intent prediction
        sup_loss, rl_loss, nlu_joint_prob, reward = criterion(
            # logits.cpu().unsqueeze(1).expand(-1, sample_size, -1),
            slot_logits.cpu().unsqueeze(1),#.expand(-1, sample_size, -1),
            # targets.cpu().clone().detach(),
            batch['labels'].clone().cpu(),
            decisions=samples.cpu(),
            intent_logits=intent_logits.cpu().unsqueeze(1) if self.with_intent else None,#.expand(-1, sample_size, -1),
            intent_targets=batch['intent'].cpu() if self.with_intent else None,
            n_supervise=1,
            calculate_reward=reinforce,
            # inputs=inputs
        )
        # pdb.set_trace()
        
        has_rl = isinstance(rl_loss, torch.Tensor)

        if not testing:
            if supervised and isinstance(sup_loss, torch.Tensor):
                # sup_loss.backward(retain_graph=(retain_graph or has_rl))
                sup_loss.backward(retain_graph=(retain_graph or has_rl or self.dim_loss))
                # pdb.set_trace()
            if reinforce and has_rl:
                rl_loss.backward(retain_graph=retain_graph)
            if optimize and not self.dim_loss:
                if max_norm:
                    clip_grad_norm_(self.nlu_parameters, max_norm)
                self.nlu_optimizer.step()
                self.nlu_optimizer.zero_grad()

        if testing and result_path:
            self._record_nlu_test_result(
                result_path,
                encoder_input,
                decoder_label,
                prediction
            )
        # print(sup_loss, rl_loss)
        # return sup_loss + rl_loss, logits, samples, nlu_joint_prob, reward
        return sup_loss + rl_loss, slot_logits, outputs_indices, intent_logits, intent_prediction, samples, nlu_joint_prob, reward

    def run_test_nlg_batch(self, batch, criterion, scorer=None,
                      testing=False, optimize=True, teacher_forcing_ratio=0.5,
                      max_norm=None, retain_graph=False, result_path=None,
                      beam_size=1, nlg_st=True, supervised=True, reinforce=False):
        if testing:
            self.nlg.eval()
        else:
            self.nlg.train()

        """
        return {
            'slot_key':flat_keys,
            'slot_key_lens':input_term_lens,
            'slot_value':flat_values,
            'slot_value_lens':input_lens,
            'intent':intent,
            'target':target,
            'multi_refs': multi_refs
            }

        bos = dataset.tokenizer.token_to_id('[CLS]')
        attrs = (b['slot_key'], b['slot_key_lens'], b['slot_value'], b['slot_value_lens'], b['intent'])
        # logits, outputs, decisions = model.forward(attrs, bos, b['target'])
        logits, outputs, decisions = model.forward(attrs, bos, b['target'], beam_size=1)
        """
        # return {'slot_key':flat_keys, 'slot_key_lens':input_term_lens, 'slot_value':flat_values, 'slot_value_lens':input_lens, 'intent':intent, 'target':target}
        # encoder_input, decoder_label, refs, sf_data = batch

        # attrs = self._sequences_to_nhot(encoder_input, self.attr_vocab_size)
        # attrs = torch.from_numpy(attrs).to(self.device)
        # bos = self.data_engine.tokenizer.token_to_id('[CLS]')
        # intent = batch['intent'].clone().detach().to(self.device) if batch['intent'] else torch.tensor([]).to(self.device)
        
        attrs = (
            batch['slot_key'].clone().detach().to(self.device),
            torch.tensor(batch['slot_key_lens']).to(self.device),
            batch['slot_value'].clone().detach().to(self.device),
            torch.tensor(batch['slot_value_lens']).to(self.device),
            batch['intent'].clone().detach().to(self.device))
        # labels = torch.from_numpy(decoder_label).to(self.device)
        labels = batch['target'].clone().detach().to(self.device)
        refs = batch['multi_refs']

        # logits.size() == (batch_size, beam_size, seq_length, vocab_size)
        # outputs.size() == (batch_size, beam_size, seq_length, vocab_size) one-hot vectors
        # Note that outputs are still in computational graph
        logits, outputs, decisions, semantic_embs = self.nlg(
            attrs,
            _BOS,
            labels, 
            beam_size=beam_size,
            tf_ratio=teacher_forcing_ratio if not testing else 0.0,
            st=nlg_st
        )

        batch_size, _, seq_length, vocab_size = logits.size()

        outputs_indices = decisions[:, 0].detach().cpu().clone().numpy()
        outputs_indices = np.argmax(outputs_indices, axis=-1)
        if scorer:
            labels_clone = labels.detach().cpu().numpy()
            scorer.update(labels_clone, refs, outputs_indices)

        # pdb.set_trace()

        sup_loss, rl_loss, nlg_joint_prob, reward = criterion(
            logits.cpu(),
            labels.cpu(),
            decisions=decisions.cpu(),
            n_supervise=1,
            calculate_reward=reinforce,
            # inputs=attrs
        )
        has_rl = isinstance(rl_loss, torch.Tensor)
        # pdb.set_trace()
        if not testing:
            if supervised and isinstance(sup_loss, torch.Tensor):
                # pdb.set_trace()
                sup_loss.backward(retain_graph=(retain_graph or has_rl or self.dim_loss))
            if reinforce and has_rl:
                rl_loss.backward(retain_graph=retain_graph)
            if optimize and not self.dim_loss:
                if max_norm:
                    clip_grad_norm_(self.nlg_parameters, max_norm)
                self.nlg_optimizer.step()
                self.nlg_optimizer.zero_grad()
                # self.nlu_optimizer.zero_grad()

        """
        if testing and result_path:
            self._record_nlg_test_result(
                result_path,
                encoder_input,
                decoder_label,
                sf_data,
                outputs_indices
            )
        """

        return sup_loss + rl_loss, logits, outputs, nlg_joint_prob, reward, attrs, semantic_embs

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

    def _sample_nlu_output(self, logits, sample_size=1):
        """
        args:
            logits: tensor of shape (batch_size, vocab_size), unnormalized logits
        returns:
            samples: tensor of shape (batch_size, sample_size, vocab_size), 0/1 decisions
        """
        y_soft = logits.sigmoid()
        y_soft_clone = y_soft.detach().cpu().clone().numpy()
        samples = []
        for i in range(sample_size):
            sample = torch.tensor([
                [random.random() < y_soft_clone[b, v] for v in range(y_soft_clone.shape[1])]
                for b in range(y_soft_clone.shape[0])
            ], dtype=torch.float, device=logits.device)
            samples.append(sample)
        y_hard = torch.stack(samples, dim=0).transpose(0, 1)
        y_soft = y_soft.unsqueeze(1).expand(-1, sample_size, -1)
        return y_hard - y_soft.detach() + y_soft

    def _st_sigmoid(self, logits, hard=False, argmax=False):
        """
        args:
            logits: tensor of shape (*, vocab_size), unnormalized logits
            hard: boolean, whether to return one-hot decisions, or probabilities.
        returns:
            decisions: tensor of shape (*, vocab_size), 0/1 decisions
        """
        y_soft = logits.sigmoid()

        if hard:
            # Straight through.
            if argmax:
                y_hard = (logits == torch.max(logits, dim=2, keepdim=True)[0]).float()
            else:
                y_hard = (y_soft >= 0.5).float()
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret


    def _record_log(self,
                    epoch,
                    testing,
                    nlu_loss=None,
                    nlg_loss=None,
                    nlu_scorer=None,
                    nlg_scorer=None):
        filename = self.valid_log_path if testing else self.train_log_path
        nlu_loss = 'None' if nlu_loss is None else '{:.4f}'.format(nlu_loss)
        nlg_loss = 'None' if nlg_loss is None else '{:.3f}'.format(nlg_loss)
        if nlu_scorer is not None:
            # micro_f1, _ = nlu_scorer.get_avg_scores()
            intent_acc, slot_f1_p_r = nlu_scorer.get_avg_scores()
            intent_acc = '{:.4f}'.format(intent_acc)
            # slot_f1_p_r: [F1, precision, recall]
            slot_f1 = '{:.4f}'.format(slot_f1_p_r[0])
        else:
            slot_f1 = '-1.0'
        if nlg_scorer is not None:
            # multiple references bleu and rouge
            _, bleu, _, rouge, _, _ = nlg_scorer.get_avg_scores()
            bleu = '{:.4f}'.format(bleu)
            rouge = ' '.join(['{:.4f}'.format(s) for s in rouge])
        else:
            bleu, rouge = '-1.0', '-1.0 -1.0 -1.0'
        with open(filename, 'a') as file:
            # file.write("{},{},{},{},"
            #            "{},{}\n".format(epoch, nlu_loss, nlg_loss, micro_f1, bleu, rouge))
            file.write(f"{epoch},{nlu_loss},{nlg_loss},{intent_acc},{slot_f1},{bleu},{rouge}\n")

    def _record_nlu_test_result(self,
                                result_path,
                                encoder_input,
                                decoder_label,
                                prediction):
        pass

    def _record_nlg_test_result(self, result_path, encoder_input,
                            decoder_label, sf_data, decoder_result):
        encoder_input = [
            # self.data_engine.tokenizer.untokenize(sent, sf_data[idx], is_token=True)
            self.data_engine.untokenize(sent)
            for idx, sent in enumerate(encoder_input)
        ]
        decoder_label = [
            # self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            self.data_engine.untokenize(sent)
            for idx, sent in enumerate(decoder_label)
        ]
        decoder_result = [
            # self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            self.data_engine.untokenize(sent)
            for idx, sent in enumerate(decoder_result)
        ]

        with open(result_path, 'a') as file:
            for idx in range(len(encoder_input)):
                file.write("---------\n")
                file.write("Data {}\n".format(idx))
                file.write("encoder input: {}\n".format(' '.join(encoder_input[idx])))
                file.write("decoder output: {}\n".format(' '.join(decoder_result[idx])))
                file.write("decoder label: {}\n".format(' '.join(decoder_label[idx])))
