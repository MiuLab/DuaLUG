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

from module_dual import NLURNN, NLGRNN, LMRNN
from utils import *
from text_token import _UNK, _PAD, _BOS, _EOS
from model_utils import collate_fn_nlg, collate_fn_nlu, collate_fn_nl, collate_fn_sf, build_optimizer, get_device
from logger import Logger
from data_engine_nlu_nlg import DataEngineSplit

from tqdm import tqdm
import pdb


class Dual:
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
            model='nlu-nlg',
            schedule='iterative',
            device=None,
            dir_name='test',
            f1_per_sample=False,
            with_intent=True#,
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
        self.model = model
        self.schedule = schedule
        self.f1_per_sample = f1_per_sample
        self.batch_size = batch_size
        self.with_intent = with_intent
        self.device = get_device(device)

        """
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
        """
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


        self.model_dir, self.log_dir = handle_model_dirs(
            model_dir, log_dir, dir_name, replace_model, is_load
        )

        if is_load:
            self.load_model(self.model_dir)

        if schedule == 'semi':
            data_size = len(train_data_engine)
            n_labeled = int(data_size * 0.03)

            train_labeled_data_engine = DataEngineSplit(
                train_data_engine.input_data[:n_labeled],
                train_data_engine.output_labels[:n_labeled],
                train_data_engine.refs[:n_labeled],
                train_data_engine.sf_data[:n_labeled],
                train_data_engine.input_attr_seqs[:n_labeled]
            )
            train_unlabeled_data_engine = DataEngineSplit(
                train_data_engine.input_data[n_labeled:],
                train_data_engine.output_labels[n_labeled:],
                train_data_engine.refs[n_labeled:],
                train_data_engine.sf_data[n_labeled:],
                train_data_engine.input_attr_seqs[n_labeled:]
            )

            self.train_nlu_data_loader = DataLoader(
                    train_labeled_data_engine,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    drop_last=True,
                    collate_fn=collate_fn_nlu,
                    pin_memory=True)

            self.train_nlu_unlabeled_data_loader = DataLoader(
                    train_unlabeled_data_engine,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    drop_last=True,
                    collate_fn=collate_fn_nlu,
                    pin_memory=True)

            self.train_nlg_data_loader = DataLoader(
                    train_labeled_data_engine,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    drop_last=True,
                    collate_fn=collate_fn_nlg,
                    pin_memory=True)

            self.train_nlg_unlabeled_data_loader = DataLoader(
                    train_unlabeled_data_engine,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    drop_last=True,
                    collate_fn=collate_fn_nlg,
                    pin_memory=True)
        else:
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

        # Initialize data loaders and optimizers
        self.train_data_engine = train_data_engine
        self.test_data_engine = test_data_engine

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
        
        print_time_info("Model create completed.")

        self.train_log_path = os.path.join(self.log_dir, "train_log.csv")
        self.valid_log_path = os.path.join(self.log_dir, "valid_log.csv")

        self.test_result_path = os.path.join(self.log_dir, "test_result.txt")

        with open(self.train_log_path, 'w') as file:
            file.write("epoch,nlu_loss,nlg_loss,intent_acc,slot_f1,bleu,rouge(1,2,L)\n")
        with open(self.valid_log_path, 'w') as file:
            file.write("epoch,nlu_loss,nlg_loss,intent_acc,slot_f1,bleu,rouge(1,2,L)\n")

        # Initialize batch count
        self.nlu_batches = self.nlg_batches = 0

    def train(self, epochs, batch_size, criterion_nlu, criterion_nlg,
              verbose_epochs=1, verbose_batches=1,
              valid_epochs=1, valid_batches=1000,
              save_epochs=10,
              teacher_forcing_ratio=0.5,
              tf_decay_rate=0.9,
              max_norm=0.25,
              mid_sample_size=1,
              dual_sample_size=1,
              nlu_st=True,
              nlg_st=True,
              primal_supervised=True,
              dual_supervised=True,
              primal_reinforce=False,
              dual_reinforce=True,
              single_direction=0):

        if mid_sample_size > 1 and dual_sample_size > 1:
            raise ValueError("mid_sample_size > 1 and dual_sample_size > 1 "
                             "is not allowed")

        self.nlu_batches = self.nlg_batches = 0

        def train_nlu():
            epoch_nlu_loss = 0
            batch_amount_nlu = 0
            scorer = IntentPredSlotFillScorer(intent_acc=self.with_intent)
            criterion_nlu.set_scorer(scorer)
            pbar = tqdm(self.train_nlu_data_loader, dynamic_ncols=True)
            for b_idx, batch in enumerate(pbar):
                self.nlu_batches += 1
                batch_loss, batch_logits, _,intent_logits, _, _, _, _ = self.run_nlu_batch(
                        batch,
                        criterion_nlu,
                        scorer=scorer,
                        testing=False,
                        max_norm=max_norm,
                        sample_size=mid_sample_size,
                        supervised=primal_supervised,
                        reinforce=primal_reinforce
                )
                epoch_nlu_loss += batch_loss.item()
                batch_amount_nlu += 1
                pbar.set_postfix(ULoss="{:.5f}".format(epoch_nlu_loss / batch_amount_nlu))

            scorer.print_avg_scores()
            return epoch_nlu_loss / batch_amount_nlu, scorer

        def train_nlg():
            epoch_nlg_loss = 0
            batch_amount_nlg = 0
            scorer = SequenceScorer()
            criterion_nlg.set_scorer(scorer)
            pbar = tqdm(self.train_nlg_data_loader, dynamic_ncols=True)
            for b_idx, batch in enumerate(pbar):
                self.nlg_batches += 1
                batch_loss, batch_logits, batch_decode_result, _, _, _, _ = self.run_nlg_batch(
                        batch,
                        criterion_nlg,
                        scorer=scorer,
                        testing=False,
                        teacher_forcing_ratio=teacher_forcing_ratio,
                        max_norm=max_norm,
                        beam_size=mid_sample_size,
                        supervised=primal_supervised,
                        reinforce=primal_reinforce
                )
                epoch_nlg_loss += batch_loss.item()
                batch_amount_nlg += 1
                pbar.set_postfix(GLoss="{:.5f}".format(epoch_nlg_loss / batch_amount_nlg))

            scorer.print_avg_scores()
            return epoch_nlg_loss / batch_amount_nlg, scorer

        def train_joint():
            nlg_loss = nlu_loss = 0
            nlg_loss_gen = nlu_loss_gen = 0
            batch_amount = 0
            nlu_scorer = IntentPredSlotFillScorer(intent_acc=self.with_intent)
            nlu_scorer_gen = IntentPredSlotFillScorer(intent_acc=self.with_intent)
            nlg_scorer = SequenceScorer()
            nlg_scorer_gen = SequenceScorer()
            criterion_nlg.set_scorer(nlg_scorer)
            criterion_nlu.set_scorer(nlu_scorer)
            
            pbar = tqdm(
                zip(self.train_nlg_data_loader, self.train_nlu_data_loader),
                total=len(self.train_nlg_data_loader),
                dynamic_ncols=True
            )

            for batch_nlg, batch_nlu in pbar:
                """
                x: semantics, y: natural language
                x_y_hat_pair = (input_semantics, batch_logits)
                x_y_hat_pair[0].size() torch.Size([64, 79])
                x_y_hat_pair[0].unsqueeze(2)

                x_y_hat_pair[1].size() torch.Size([64, 1, 37, 504])
                shape [batch_size, beam_size, seq_length, vocab_size]
                x_y_hat_pair[1].squeeze().transpose(1, 2)
                """
                """
                batch_loss_nlg, batch_loss_nlu, y_x_hat_pair, x_hat_y_hat_pair = train_nlu_nlg_joint_batch(
                    batch_nlu,
                    nlu_scorer=nlu_scorer,
                    nlg_scorer=nlg_scorer_gen)
                """
                # '''
                # NLG -> NLU
                batch_loss_nlg, batch_loss_nlu, x_y_hat_pair, y_hat_x_hat_pair = train_nlg_nlu_joint_batch(
                    batch_nlg,
                    nlu_scorer=nlu_scorer_gen,
                    nlg_scorer=nlg_scorer)
                nlg_loss += batch_loss_nlg
                nlu_loss_gen += batch_loss_nlu

                if not single_direction:
                    # NLU -> NLG
                    batch_loss_nlg, batch_loss_nlu = train_nlu_nlg_joint_batch(
                        batch_nlu,
                        nlu_scorer=nlu_scorer,
                        nlg_scorer=nlg_scorer_gen)

                if max_norm:
                    clip_grad_norm_(self.nlg_parameters, max_norm)
                    clip_grad_norm_(self.nlu_parameters, max_norm)

                self.nlu_optimizer.step()
                self.nlg_optimizer.step()
                self.nlu_optimizer.zero_grad()
                self.nlg_optimizer.zero_grad()
                
                nlg_loss_gen += batch_loss_nlg
                nlu_loss += batch_loss_nlu
                batch_amount += 1

                pbar.set_postfix(
                        UT="{:.4f}".format(nlu_loss / batch_amount),
                        UF="{:.4f}".format(nlu_loss_gen / batch_amount),
                        GT="{:.3f}".format(nlg_loss / batch_amount),
                        GF="{:.3f}".format(nlg_loss_gen / batch_amount)
                )
                # break

            print_time_info("True NLG scores:")
            nlg_scorer.print_avg_scores()
            print_time_info("Generated NLG scores:")
            nlg_scorer_gen.print_avg_scores()

            print_time_info("True NLU scores:")
            nlu_scorer.print_avg_scores()
            print_time_info("Generated NLU scores:")
            nlu_scorer_gen.print_avg_scores()

            return (
                nlg_loss / batch_amount,
                nlg_loss_gen / batch_amount,
                nlu_loss / batch_amount,
                nlu_loss_gen / batch_amount,
                nlg_scorer,
                nlg_scorer_gen,
                nlu_scorer,
                nlu_scorer_gen
            )

        def train_nlg_nlu_joint_batch(batch, nlu_scorer=None, nlg_scorer=None):
            criterion_nlg.set_scorer(nlg_scorer)
            criterion_nlu.set_scorer(nlu_scorer)
            

            """
            return {
                'slot_key':flat_keys,
                'slot_key_lens':input_term_lens,
                'slot_value':flat_values,
                'slot_value_lens':input_lens,
                'intent':intent,
                'target':target,
                'multi_refs': multi_refs,
                'dual_nlu_inputs': inputs,
                'dual_nlu_labels': labels,
                'dual_nlu_intent': intent
            }
            """
            attrs = (
                batch['slot_key'],
                batch['slot_key_lens'],
                batch['slot_value'],
                batch['slot_value_lens'],
                batch['intent'])
            refs = batch['multi_refs']
            
            self.nlg_batches += 1
            batch_loss_nlg, batch_logits, batch_decode_result, \
                nlg_joint_prob, last_reward, input_semantics, semantic_embs = self.run_nlg_batch(
                    batch,
                    criterion_nlg,
                    scorer=nlg_scorer,
                    testing=False,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    max_norm=max_norm,
                    retain_graph=True,
                    optimize=False,
                    beam_size=mid_sample_size,
                    nlg_st=nlg_st,
                    supervised=primal_supervised,
                    reinforce=primal_reinforce
            )
            # pdb.set_trace()
            x_y_hat_pair = (semantic_embs.transpose(0,1), batch_logits)

            # generated_batch = [batch_decode_result, encoder_input, refs, sf_data]
            generated_batch = [
                torch.argmax(batch_decode_result, dim=3).squeeze(),
                batch['dual_nlu_labels'],
                batch['dual_nlu_intent']]
            
            self.nlu_batches += 1
            # return sup_loss + rl_loss, slot_logits, outputs_indices, intent_logits, intent_prediction, samples, nlu_joint_prob, reward
            batch_loss_nlu, batch_logits, slot_prediction, intent_logits, intent_prediction, _, _, _ = self.run_nlu_batch_dual(
                    generated_batch,
                    criterion_nlu,
                    scorer=nlu_scorer,
                    joint_prob_other=nlg_joint_prob if mid_sample_size > 1 else None,
                    max_norm=max_norm,
                    last_reward=last_reward,
                    sample_size=dual_sample_size,
                    supervised=dual_supervised,
                    reinforce=dual_reinforce,
                    teacher_forcing_ratio=teacher_forcing_ratio,
            )
            
            semantic_frames = list()
            for slot_seq, word_seq in zip(slot_prediction, generated_batch[0].clone().cpu().numpy()):
                semantic_frames.append(self.train_data_engine.fuzzy_mapping_slots_to_semantic_frame(slot_seq, word_seq))
            nlg_inputs = self.train_data_engine.batch_semantic_frame_to_nlg_input(semantic_frames)
            
            attrs = (
                nlg_inputs['slot_key'].clone().detach().to(self.device),
                torch.tensor(nlg_inputs['slot_key_lens']).to(self.device),
                nlg_inputs['slot_value'].clone().detach().to(self.device),
                torch.tensor(nlg_inputs['slot_value_lens']).to(self.device),
                torch.tensor(intent_prediction).to(self.device))
            semantic_embs = self.nlg.encode_slot_intent(*attrs).unsqueeze(1)
            y_hat_x_hat_pair = (x_y_hat_pair[1], semantic_embs)

            return batch_loss_nlg.item(), batch_loss_nlu.item(), x_y_hat_pair, y_hat_x_hat_pair

        def train_nlu_nlg_joint_batch(batch, nlu_scorer=None, nlg_scorer=None):
            criterion_nlg.set_scorer(nlg_scorer)
            criterion_nlu.set_scorer(nlu_scorer)
            encoder_input = batch['inputs']

            self.nlu_batches += 1
            batch_loss_nlu, batch_logits, slot_prediction, intent_logits, intent_prediction, samples, \
                nlu_joint_prob, last_reward = self.run_nlu_batch(
                    batch,
                    criterion_nlu,
                    scorer=nlu_scorer,
                    testing=False,
                    max_norm=max_norm,
                    retain_graph=True,
                    optimize=False,
                    beam_size=mid_sample_size,
                    sample_size=mid_sample_size,
                    supervised=primal_supervised,
                    reinforce=primal_reinforce
            )

            y = torch.zeros(self.batch_size, encoder_input.size()[1], self.vocab_size).scatter_(2, encoder_input.unsqueeze(2),1).to(self.device)

            # convert nlu output to nlg input
            semantic_frames = list()
            for slot_seq, word_seq in zip(slot_prediction, batch['inputs'].clone().numpy()):
                semantic_frames.append(self.train_data_engine.fuzzy_mapping_slots_to_semantic_frame(slot_seq, word_seq))
            nlg_inputs = self.train_data_engine.batch_semantic_frame_to_nlg_input(semantic_frames)
            # x_hat
            attrs = (
                nlg_inputs['slot_key'].clone().detach().to(self.device),
                torch.tensor(nlg_inputs['slot_key_lens']).to(self.device),
                nlg_inputs['slot_value'].clone().detach().to(self.device),
                torch.tensor(nlg_inputs['slot_value_lens']).to(self.device),
                torch.tensor(intent_prediction).to(self.device))
            semantic_embs = self.nlg.encode_slot_intent(*attrs).unsqueeze(1)

            y_x_hat_pair = (y, semantic_embs)

            generated_batch = [attrs, encoder_input, batch['dual_nlg_multi_refs']]

            self.nlg_batches += 1
            batch_loss_nlg, batch_logits, batch_decode_result, _, _, _ = self.run_nlg_batch_dual(
                    generated_batch,
                    criterion_nlg,
                    scorer=nlg_scorer,
                    joint_prob_other=nlu_joint_prob if mid_sample_size > 1 else None,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    max_norm=max_norm,
                    last_reward=last_reward,
                    beam_size=dual_sample_size,
                    supervised=dual_supervised,
                    reinforce=dual_reinforce
            )

            return batch_loss_nlg.item(), batch_loss_nlu.item()#, y_x_hat_pair, x_hat_y_hat_pair

        def train_semi(epoch):
            nlu_sup_loss, nlu_sup_scorer = train_nlu()
            nlg_sup_loss, nlg_sup_scorer = train_nlg()

            pbar = tqdm(
                zip(self.train_nlg_unlabeled_data_loader, self.train_nlu_unlabeled_data_loader),
                total=len(self.train_nlg_unlabeled_data_loader),
                dynamic_ncols=True
            )

            for bid, (batch_nlg, batch_nlu) in enumerate(pbar):
                if bid == 5:
                    break
                batch_loss_nlg, batch_loss_nlu_dual = train_nlg_nlu_semi_batch(batch_nlg)
                batch_loss_nlg_dual, batch_loss_nlu = train_nlu_nlg_semi_batch(batch_nlu)
                pbar.set_postfix(
                        UT="{:.4f}".format(batch_loss_nlu),
                        UF="{:.4f}".format(batch_loss_nlu_dual),
                        GT="{:.3f}".format(batch_loss_nlg),
                        GF="{:.3f}".format(batch_loss_nlg_dual)
                )

            return nlg_sup_loss, nlg_sup_scorer, nlu_sup_loss, nlu_sup_scorer

        def train_nlg_nlu_semi_batch(batch):
            encoder_input, decoder_label, refs, sf_data = batch
            self.nlg_batches += 1
            batch_loss_nlg, batch_logits, batch_decode_result, \
                nlg_joint_prob, last_reward = self.run_nlg_batch(
                    batch,
                    criterion_nlg,
                    testing=False,
                    teacher_forcing_ratio=0.0,
                    max_norm=max_norm,
                    retain_graph=True,
                    optimize=False,
                    beam_size=mid_sample_size,
                    nlg_st=True,
                    supervised=False,
                    reinforce=True
            )
            generated_batch = [batch_decode_result, encoder_input, refs, sf_data]
            self.nlu_batches += 1
            batch_loss_nlu, batch_logits, _, _, _ = self.run_nlu_batch_dual(
                    generated_batch,
                    criterion_nlu,
                    joint_prob_other=nlg_joint_prob if mid_sample_size > 1 else None,
                    max_norm=max_norm,
                    supervised=True,
                    reinforce=False
            )

            return batch_loss_nlg.item(), batch_loss_nlu.item()

        def train_nlu_nlg_semi_batch(batch):
            encoder_input, decoder_label, refs, sf_data = batch
            self.nlu_batches += 1
            batch_loss_nlu, batch_logits, samples, \
                nlu_joint_prob, last_reward = self.run_nlu_batch(
                    batch,
                    criterion_nlu,
                    testing=False,
                    max_norm=max_norm,
                    retain_graph=True,
                    optimize=False,
                    sample_size=mid_sample_size,
                    supervised=False,
                    reinforce=True
            )

            generated_batch = [samples, encoder_input, refs, sf_data]

            self.nlg_batches += 1
            batch_loss_nlg, batch_logits, batch_decode_result, _, _ = self.run_nlg_batch_dual(
                    generated_batch,
                    criterion_nlg,
                    joint_prob_other=nlu_joint_prob if mid_sample_size > 1 else None,
                    teacher_forcing_ratio=0.0,
                    max_norm=max_norm,
                    supervised=True,
                    reinforce=False
            )

            return batch_loss_nlg.item(), batch_loss_nlu.item()

        nlu_loss = nlg_loss = None
        nlu_scorer = nlg_scorer = None
        test_nlu = test_nlg = True
        for idx in range(1, epochs+1):
            if self.model == "nlu":
                test_nlg = False
                nlu_loss, nlu_scorer = train_nlu()
            elif self.model == "nlg":
                test_nlu = False
                nlg_loss, nlg_scorer = train_nlg()
            elif self.model == "nlu-nlg":
                if self.schedule == "iterative":
                    nlu_loss, nlu_scorer = train_nlu()
                    nlg_loss, nlg_scorer = train_nlg()
                elif self.schedule == "joint":
                    nlg_loss, _, nlu_loss, _, nlg_scorer, _, \
                        nlu_scorer, _ = train_joint()
                elif self.schedule == "semi":
                    nlg_loss, nlg_scorer, nlu_loss, nlu_scorer = train_semi(idx)

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
                test_nlu=test_nlu,
                test_nlg=test_nlg,
                epoch=idx
            )

            criterion_nlu.epoch_end()
            criterion_nlg.epoch_end()

            teacher_forcing_ratio *= tf_decay_rate

    def test_dump(self, criterion_nlu, criterion_nlg):
        with torch.no_grad():
            with open(os.path.join(self.log_dir, 'validation', 'test_dump_NSN.txt'), 'w') as fw:
                data_count = 0
                nlg_loss = 0
                batch_amount = 0
                for b_idx, batch in enumerate(tqdm(self.test_nlu_data_loader)):
                    batch_amount += 1
                    encoder_input, decoder_label, refs, sf_data = batch
                    _, _, pred, _, _ = self.run_nlu_batch(
                        batch,
                        criterion_nlu,
                        testing=True,
                    )

                    generated_batch = [pred, encoder_input, refs, sf_data]

                    loss, _, decode_result, _, _ = self.run_nlg_batch_dual(
                            generated_batch,
                            criterion_nlg,
                            teacher_forcing_ratio=0.0
                    )
                    nlg_loss += loss.item()

                    for i, (x, y_true, y_pred, x_pred) in enumerate(zip(
                            encoder_input,
                            decoder_label,
                            pred.cpu().long().numpy()[:, 0],
                            decode_result.cpu().long().numpy()[:, 0])):
                        x = self.data_engine.untokenize(x)
                        y_true = self.data_engine.untokenize(sorted(y_true))
                        y_pred = self.data_engine.untokenize(sorted(np.where(y_pred==1)[0]))
                        x_pred = self.data_engine.untokenize(np.argmax(x_pred, axis=-1))
                        
                        fw.write("Data {}\n".format(data_count+i))
                        fw.write("NL input: {}\n".format(" ".join(x)))
                        fw.write("SF label: {}\n".format(" / ".join(y_true)))
                        fw.write("SF pred: {}\n".format(" / ".join(y_pred)))
                        fw.write("NL output: {}\n\n".format(" ".join(x_pred)))
                    data_count += len(encoder_input)
                print('nlg reconstruction loss {}'.format(nlg_loss/batch_amount))
            with open(os.path.join(self.log_dir, 'validation', 'test_dump_SNS.txt'), 'w') as fw:
                data_count = 0
                nlu_loss = 0
                batch_amount = 0
                for b_idx, batch in enumerate(tqdm(self.test_nlg_data_loader)):
                    batch_amount += 1
                    encoder_input, decoder_label, refs, sf_data = batch
                    _, _, decode_result, _, _, _ = self.run_nlg_batch(
                        batch,
                        criterion_nlg,
                        teacher_forcing_ratio=0.0,
                        testing=True,
                    )

                    generated_batch = [decode_result, encoder_input, refs, sf_data]

                    loss, _, pred, _, _ = self.run_nlu_batch_dual(
                            generated_batch,
                            criterion_nlu,
                    )
                    nlu_loss += loss.item()

                    for i, (x, y_true, y_pred, x_pred) in enumerate(zip(
                            encoder_input,
                            decoder_label,
                            decode_result.cpu().long().numpy()[:, 0],
                            pred.cpu().long().numpy()[:, 0])):
                        x = self.data_engine.untokenize(sorted(x))
                        y_true = self.data_engine.untokenize(y_true)
                        y_pred = self.data_engine.untokenize(np.argmax(y_pred, axis=-1))
                        x_pred = self.data_engine.untokenize(sorted(np.where(x_pred==1)[0]))
                        fw.write("Data {}\n".format(data_count+i))
                        fw.write("SF input: {}\n".format(" / ".join(x)))
                        fw.write("NL label: {}\n".format(" ".join(y_true)))
                        fw.write("NL pred: {}\n".format(" ".join(y_pred)))
                        fw.write("SF output: {}\n\n".format(" / ".join(x_pred)))
                    data_count += len(encoder_input)
                print('nlu reconstruction loss: {}'.format(nlu_loss/batch_amount))

    def test(self, batch_size,
             criterion_nlu, criterion_nlg,
             test_nlu=True, test_nlg=True,
             sample_size=1, epoch=-1):

        nlu_loss = nlg_loss = None
        nlu_scorer = nlg_scorer = None

        batch_amount = 0

        if test_nlu:
            nlu_scorer = IntentPredSlotFillScorer(intent_acc=self.with_intent)
            nlu_loss = 0
            for b_idx, batch in enumerate(tqdm(self.test_nlu_data_loader)):
                with torch.no_grad():
                    batch_loss, batch_logits, slot_prediction, intent_logits, intent_prediction, _, _, _ = self.run_nlu_batch(
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
                    batch_loss, batch_logits, batch_decode_result, _, _, _, _ = self.run_nlg_batch(
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

        if test_nlg:
            nlg_scorer.write_avg_scores_to_file(self.test_result_path)
        if test_nlu:
            nlu_scorer.write_avg_scores_to_file(self.test_result_path)

    def run_nlu_batch(self, batch, criterion, scorer=None,
                      testing=False, optimize=True, max_norm=None, beam_size=1,
                      teacher_forcing_ratio=0.5, retain_graph=False, result_path=None, sample_size=1,
                      supervised=True, reinforce=False):
        if testing:
            self.nlu.eval()
        else:
            self.nlu.train()

        inputs = batch['inputs'].to(self.device)
        targets = batch['labels'].clone().detach().to(self.device)

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

        if sample_size > 1:
            samples = self._sample_nlu_output(logits, sample_size)
        else:
            # samples = self._st_sigmoid(logits, hard=True).unsqueeze(1)
            samples = self._st_sigmoid(slot_logits, hard=True, argmax=True).unsqueeze(1)

        # loss for slot filling and intent prediction
        sup_loss, rl_loss, nlu_joint_prob, reward = criterion(
            slot_logits.cpu().unsqueeze(1),#.expand(-1, sample_size, -1),
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
                sup_loss.backward(retain_graph=(retain_graph or has_rl))
            if reinforce and has_rl:
                rl_loss.backward(retain_graph=retain_graph)
            if optimize:
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
        return sup_loss + rl_loss, slot_logits, outputs_indices, intent_logits, intent_prediction, samples, nlu_joint_prob, reward

    def run_nlu_batch_dual(self, batch, criterion, scorer=None,
                           max_norm=None, joint_prob_other=None,
                           last_reward=0.0, sample_size=1, supervised=True, 
                           reinforce=True, teacher_forcing_ratio=0.9):
        self.nlu.train()

        encoder_input, decoder_label, intent_label = batch

        inputs = encoder_input.to(self.device)
        sampled_input = (inputs.size(1) > 1)

        targets = decoder_label.clone().detach().to(self.device)

        slot_logits, outputs, decisions, intent_logits = self.nlu(
            inputs,
            _BOS,
            labels=targets, 
            beam_size=sample_size,
            tf_ratio=teacher_forcing_ratio)

        outputs_indices = decisions[:, 0].detach().cpu().clone().numpy()
        outputs_indices = np.argmax(outputs_indices, axis=-1)
        slot_prediction = outputs_indices
        intent_prediction = torch.argmax(intent_logits.detach().cpu(), dim=1)
        intent_prediction = intent_prediction.clone().numpy()
        
        samples = decisions

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
                intent_labels=intent_label.clone().cpu().numpy() if self.with_intent else None,
                intent_prediction=intent_prediction if self.with_intent else None
            )

        sup_loss, rl_loss, nlu_joint_prob, reward = criterion(
            slot_logits.cpu().unsqueeze(1),
            targets.clone().cpu(),
            decisions=samples.cpu(),
            intent_logits=intent_logits.cpu().unsqueeze(1) if self.with_intent else None,#.expand(-1, sample_size, -1),
            intent_targets=intent_label.cpu() if self.with_intent else None,
            n_supervise=1,
            calculate_reward=reinforce,
        )
        
        has_rl = isinstance(rl_loss, torch.Tensor)

        if supervised and isinstance(sup_loss, torch.Tensor):
            sup_loss.backward(retain_graph=has_rl)
        if reinforce and has_rl:
            rl_loss.backward(retain_graph=False)
        if max_norm:
            clip_grad_norm_(self.nlg_parameters, max_norm)
            clip_grad_norm_(self.nlu_parameters, max_norm)

        self.nlu_optimizer.step()
        self.nlg_optimizer.step()
        self.nlu_optimizer.zero_grad()
        self.nlg_optimizer.zero_grad()


        return sup_loss + rl_loss, slot_logits, outputs_indices, intent_logits, intent_prediction, samples, nlu_joint_prob, reward

    def run_nlg_batch(self, batch, criterion, scorer=None,
                      testing=False, optimize=True, teacher_forcing_ratio=0.5,
                      max_norm=None, retain_graph=False, result_path=None,
                      beam_size=1, nlg_st=True, supervised=True, reinforce=False):
        if testing:
            self.nlg.eval()
        else:
            self.nlg.train()


        attrs = (
            batch['slot_key'].clone().detach().to(self.device),
            torch.tensor(batch['slot_key_lens']).to(self.device),
            batch['slot_value'].clone().detach().to(self.device),
            torch.tensor(batch['slot_value_lens']).to(self.device),
            batch['intent'].clone().detach().to(self.device))
        labels = batch['target'].clone().detach().to(self.device)
        refs = batch['multi_refs']

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


        sup_loss, rl_loss, nlg_joint_prob, reward = criterion(
            logits.cpu(),
            labels.cpu(),
            decisions=decisions.cpu(),
            n_supervise=1,
            calculate_reward=reinforce,
        )
        has_rl = isinstance(rl_loss, torch.Tensor)
        if not testing:
            if supervised and isinstance(sup_loss, torch.Tensor):
                sup_loss.backward(retain_graph=(retain_graph or has_rl))
            if reinforce and has_rl:
                rl_loss.backward(retain_graph=retain_graph)
            if optimize:
                if max_norm:
                    clip_grad_norm_(self.nlg_parameters, max_norm)
                self.nlg_optimizer.step()
                self.nlg_optimizer.zero_grad()

        return sup_loss + rl_loss, logits, outputs, nlg_joint_prob, reward, attrs, semantic_embs

    # WIP
    def run_nlg_batch_dual(self, batch, criterion, scorer=None,
                           joint_prob_other=None,
                           teacher_forcing_ratio=0.5,
                           max_norm=None, last_reward=0.0, beam_size=1,
                           supervised=True, reinforce=True):

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
        attrs, decoder_label, refs = batch
        
        # labels = torch.from_numpy(decoder_label).to(self.device)
        # labels = batch['target'].clone().detach().to(self.device)


        # attrs = encoder_input.to(self.device)
        # labels = torch.from_numpy(decoder_label).to(self.device)
        labels = decoder_label.to(self.device)
        # bos = self.data_engine.tokenizer.token_to_id('[CLS]')

        # TODO: check handling beam search
        # sampled_input = (attrs[0].size(1) > 1)

        # bs, ss, vs = attrs.size()
        # attrs = attrs.contiguous().view(-1, vs)
        
        # logits.size() == (batch_size, beam_size, seq_length, vocab_size)
        # outputs.size() == (batch_size, beam_size, seq_length, vocab_size) one-hot vectors
        # Note that outputs are still in computational graph
        logits, outputs, decisions, semantic_embs = self.nlg(
            attrs,
            _BOS,
            # labels.unsqueeze(1).expand(-1, ss, -1).contiguous().view(-1, labels.size(-1)),
            labels,
            beam_size=beam_size,
            tf_ratio=teacher_forcing_ratio
        )
        # pdb.set_trace()

        # TODO: check handling beam search
        '''
        if sampled_input:
            _, _, sl, vs = logits.size()
            logits = logits.view(bs, ss, sl, vs)
            outputs = outputs.view(bs, ss, sl, vs)
            decisions = decisions.view(bs, ss, sl, vs)
        '''

        batch_size, _, seq_length, vocab_size = logits.size()

        outputs_indices = decisions[:, 0].detach().cpu().clone().numpy()
        outputs_indices = np.argmax(outputs_indices, axis=-1)
        if scorer:
            labels_clone = labels.detach().cpu().numpy()
            scorer.update(labels_clone, refs, outputs_indices)

        sup_loss, rl_loss, nlg_joint_prob, reward = criterion(
            logits.cpu().contiguous(),
            labels.cpu().contiguous(),
            decisions=decisions.cpu().detach(),
            log_joint_prob=joint_prob_other,
            n_supervise=1,
            calculate_reward=reinforce,
            # inputs=attrs
        )
        has_rl = isinstance(rl_loss, torch.Tensor)
        '''
        if is_dual and has_rl:
            _, dual_rl_loss, _, _ = criterion(
                logits.cpu().detach().contiguous().view(-1, vocab_size),
                labels.cpu().contiguous().view(-1),
                outputs.cpu(),
                supervised=False,
                log_joint_prob=joint_prob_other,
                last_reward=last_reward
            )
        '''

        if supervised and isinstance(sup_loss, torch.Tensor):
            sup_loss.backward(retain_graph=has_rl)
        if reinforce and has_rl:
            rl_loss.backward(retain_graph=False)
        if max_norm:
            clip_grad_norm_(self.nlu_parameters, max_norm)
            clip_grad_norm_(self.nlg_parameters, max_norm)

        self.nlg_optimizer.step()
        self.nlu_optimizer.step()
        self.nlg_optimizer.zero_grad()
        self.nlu_optimizer.zero_grad()

        return sup_loss + rl_loss, logits, outputs, nlg_joint_prob, reward, attrs

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
            print_time_info("Load model from {} successfully".format(model_dir))

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
