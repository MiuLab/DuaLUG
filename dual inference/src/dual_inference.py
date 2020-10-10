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
import json

from module import NLURNN, NLGRNN, LMRNN
# from utils import single_BLEU, BLEU, single_ROUGE, ROUGE, best_ROUGE, print_time_info, check_dir, print_curriculum_status
from utils import *
from text_token import _UNK, _PAD, _BOS, _EOS
# from model_utils import collate_fn_nlg, collate_fn_nlu, collate_fn_nl, collate_fn_sf, build_optimizer, get_device
# from logger import Logger
# from data_engine import DataEngineSplit

from tqdm import tqdm

class DualInference():
    def __init__(self, nlu_weight=0.5, nlg_weight=0.5, lm_weight=0.1, made_weight=0.1, top_k=10, lm=None, made=None, data_engine=None):
        self.nlu_weight = nlu_weight
        self.nlg_weight = nlg_weight
        self.lm_weight = lm_weight
        self.made_weight = made_weight
        self.top_k = top_k
        self.lm = lm
        self.made = made
        self.train_data_engine = data_engine
        if self.lm is not None and self.made is not None:
            print_time_info("LM & MADE model used, support marginal probability estimating for dual inference.")
        self.device = 'cuda'

    def forward_nlu(self, inputs, slot_logits, slot_decision, intent_logits, intent_prediction, nlg_model, norm_seq_length=True):
        if self.nlu_weight < 1.0:
            nlg_model.eval()
            semantic_frames = list()
            slot_decision = slot_decision.argmax(-1)
            batch_size, sample_size, seq_length = slot_decision.shape
            for slot_seqs, word_seq in zip(slot_decision, inputs.detach().cpu().clone().numpy()):
                for slot_seq in slot_seqs:
                    semantic_frames.append(self.train_data_engine.fuzzy_mapping_slots_to_semantic_frame(slot_seq, word_seq))
            pseudo_nlg_inputs = self.train_data_engine.batch_semantic_frame_to_nlg_input(semantic_frames)
            # NLG
            beamed_intent = intent_prediction.unsqueeze(1).repeat([1, sample_size, 1]).view(sample_size*batch_size)
            pseudo_attrs = (
                pseudo_nlg_inputs['slot_key'].clone().detach().to(self.device),
                torch.tensor(pseudo_nlg_inputs['slot_key_lens']).to(self.device),
                pseudo_nlg_inputs['slot_value'].clone().detach().to(self.device),
                torch.tensor(pseudo_nlg_inputs['slot_value_lens']).to(self.device),
                beamed_intent.to(self.device))
            beamed_inputs = inputs.unsqueeze(1)
            nlg_labels = beamed_inputs.repeat([1, sample_size, 1]).view(sample_size*batch_size, -1)
            # pdb.set_trace()
            # try:
            nlg_seq_logits, nlg_outputs, semantic_embs = nlg_model.forward_greedy(
                pseudo_attrs,
                _BOS,
                nlg_labels, 
                tf_ratio=1.0,
                st=True
            )
            # except:
            # pdb.set_trace()
            # nlg_seq_logits, nlg_outputs = nlg_model.forward_greedy(
            #     samples, _BOS, inputs.repeat([sample_size, 1, 1]).view(sample_size*bs, -1),
            #     tf_ratio=1.0,
            #     st=True
            # )  # [batch_size, seq_length, vocab_size]
            _, seq_length, vocab_size = nlg_seq_logits.shape
            nlg_seq_logits = nlg_seq_logits.view(batch_size, sample_size, seq_length, vocab_size)
            # nlg_target = inputs.repeat([sample_size, 1, 1]).transpose(1, 0)
            nlg_target = nlg_labels
            nlg_log_prob = - torch.nn.functional.cross_entropy(nlg_seq_logits.contiguous().view(-1, vocab_size), nlg_target.contiguous().view(-1), ignore_index=_PAD, reduction='none')
            nlg_log_prob = nlg_log_prob.view(batch_size, sample_size, seq_length).sum(-1)
            # cands = samples.view(batch_size, sample_size, cs)
            if self.lm is not None and self.made is not None:
                # lm to estimate marginal P(X)
                lm_log_prob = self.lm.get_log_prob(beamed_inputs).cuda() # [BSZ, BEAM]
                lm_log_prob = lm_log_prob.unsqueeze(-1).repeat([1, sample_size])
                # made to estimate marginal P(Y)
                # temp_prediction = samples.detach().cpu().float()
                made_log_prob = self.made.get_log_prob(pseudo_attrs, n_samples=3).cuda().view(batch_size, sample_size)
                if norm_seq_length:
                    nlg_log_prob /= float(seq_length)
                    lm_log_prob /= float(seq_length)
                # made_slt = made_loss.argmax(-1)
                # made_logits = cands[torch.arange(0, bs), made_slt,:] #.float() * made_loss[torch.arange(0, bs), made_slt].unsqueeze(-1)
                dual_log_prob = nlg_log_prob + self.made_weight*made_log_prob.cuda() - self.lm_weight*lm_log_prob
            else:
                dual_log_prob = nlg_log_prob
            mask = (beamed_inputs != _PAD).long()
            slot_decision = slot_decision * mask + -1 * (1-mask)
            slot_log_prob = - torch.nn.functional.cross_entropy(slot_logits.contiguous().view(-1, slot_logits.size(-1)), slot_decision.contiguous().view(-1), ignore_index=-1, reduction='none')
            slot_log_prob = slot_log_prob.view(batch_size, sample_size, seq_length).sum(-1)
            # intent_log_prob =
            # pdb.set_trace()
            slot_joint_prob = (1 - self.nlu_weight) * dual_log_prob + self.nlu_weight * slot_log_prob
            slot_joint_slt = slot_joint_prob.argmax(-1)
            slot_best_prediction = slot_decision[range(batch_size), slot_joint_slt]
            ### Intent
            semantic_frames = list()
            intent_list = []
            batch_size, intent_sample_size = intent_logits.shape
            for slot_seqs, word_seq in zip(slot_best_prediction, inputs.detach().cpu().clone().numpy()):
                for i in range(intent_sample_size):
                    intent_list.append(i)
                    semantic_frames.append(self.train_data_engine.fuzzy_mapping_slots_to_semantic_frame(slot_seq, word_seq))
            pseudo_nlg_inputs = self.train_data_engine.batch_semantic_frame_to_nlg_input(semantic_frames)
            pseudo_attrs = (
                pseudo_nlg_inputs['slot_key'].clone().detach().to(self.device),
                torch.tensor(pseudo_nlg_inputs['slot_key_lens']).to(self.device),
                pseudo_nlg_inputs['slot_value'].clone().detach().to(self.device),
                torch.tensor(pseudo_nlg_inputs['slot_value_lens']).to(self.device),
                torch.tensor(intent_list).to(self.device))
            beamed_inputs = inputs.unsqueeze(1)
            nlg_labels = beamed_inputs.repeat([1, intent_sample_size, 1]).view(intent_sample_size*batch_size, -1)
            nlg_seq_logits, nlg_outputs, semantic_embs = nlg_model.forward_greedy(
                pseudo_attrs,
                _BOS,
                nlg_labels, 
                tf_ratio=1.0,
                st=True
            )
            _, seq_length, vocab_size = nlg_seq_logits.shape
            nlg_seq_logits = nlg_seq_logits.view(batch_size, intent_sample_size, seq_length, vocab_size)
            # nlg_target = inputs.repeat([sample_size, 1, 1]).transpose(1, 0)
            nlg_target = nlg_labels
            nlg_log_prob = - torch.nn.functional.cross_entropy(nlg_seq_logits.contiguous().view(-1, vocab_size), nlg_target.contiguous().view(-1), ignore_index=_PAD, reduction='none')
            nlg_log_prob = nlg_log_prob.view(batch_size, intent_sample_size, seq_length).sum(-1)
            # cands = samples.view(batch_size, sample_size, cs)
            if self.lm is not None and self.made is not None:
                # lm to estimate marginal P(X)
                lm_log_prob = self.lm.get_log_prob(beamed_inputs).cuda() # [BSZ, BEAM]
                lm_log_prob = lm_log_prob.unsqueeze(-1).repeat([1, intent_sample_size])
                # made to estimate marginal P(Y)
                # temp_prediction = samples.detach().cpu().float()
                made_log_prob = self.made.get_log_prob(pseudo_attrs, n_samples=3).cuda().view(batch_size, intent_sample_size)
                # made_slt = made_loss.argmax(-1)
                if norm_seq_length:
                    nlg_log_prob /= float(seq_length)
                    lm_log_prob /= float(seq_length)
                # made_logits = cands[torch.arange(0, bs), made_slt,:] #.float() * made_loss[torch.arange(0, bs), made_slt].unsqueeze(-1)
                dual_log_prob = nlg_log_prob + self.made_weight*made_log_prob.cuda() - self.lm_weight*lm_log_prob
            else:
                dual_log_prob = nlg_log_prob
            intent_log_prob = F.log_softmax(intent_logits, dim=-1)
            intent_joint_prob = (1 - self.nlu_weight) * dual_log_prob + self.nlu_weight * intent_log_prob
            intent_joint_slt = intent_joint_prob.argmax(-1)
            # pdb.set_trace()
            return slot_best_prediction, intent_joint_slt
        else:
            return slot_decision[:, 0, :].argmax(-1), intent_prediction

    def forward_nlg(self, attrs, logits, nlg_decisions, slot_seqs, beam_size, nlu_model, norm_seq_length=True):
        if self.nlg_weight < 1.0:
            # dual inference selection
            nlu_model.eval()
            batch_size, sample_size, seq_length, vocab_size = logits.size()
            nlg_outputs = nlg_decisions.detach().argmax(-1)  # [BSZ,BEAM,SEQ]
            nlg_outputs = nlg_outputs.view(batch_size * beam_size, seq_length)
            beamed_slot_seqs = slot_seqs.unsqueeze(1).repeat([1, beam_size, 1]).view(batch_size * beam_size, seq_length)
            slot_logits, slot_outputs, slot_decisions, intent_logits = nlu_model(
                nlg_outputs,
                _BOS,
                beamed_slot_seqs,
                beam_size=1,
                tf_ratio=1.0
            )
            # mask = (beamed_slot_seqs != _PAD).long()
            # slot_decision = slot_decision * mask + -1 * (1-mask)
            slot_log_prob = - torch.nn.functional.cross_entropy(slot_logits.contiguous().view(-1, slot_logits.size(-1)), beamed_slot_seqs.contiguous().view(-1), ignore_index=-1, reduction='none')
            slot_log_prob = slot_log_prob.view(batch_size, sample_size, seq_length).sum(-1)
            intent_targets = attrs[-1].unsqueeze(-1).repeat([1, sample_size]).view(-1)
            intent_log_prob = - torch.nn.functional.cross_entropy(intent_logits.contiguous().view(-1, intent_logits.size(-1)), intent_targets.contiguous(), reduction='none')
            intent_log_prob = intent_log_prob.view(batch_size, sample_size)
            nlu_log_prob = slot_log_prob + intent_log_prob

            if self.lm is not None and self.made is not None:
                # lm to estimate marginal P(X)
                lm_log_prob = self.lm.get_log_prob(nlg_outputs).cuda() # [BSZ, BEAM]
                lm_log_prob = lm_log_prob.view(batch_size, sample_size)
                # made to estimate marginal P(Y)
                # temp_prediction = samples.detach().cpu().float()
                made_log_prob = self.made.get_log_prob(attrs, n_samples=3).cuda()#
                made_log_prob = made_log_prob.unsqueeze(-1).repeat([1, sample_size])
                # made_slt = made_loss.argmax(-1)
                if norm_seq_length:
                    # nlg_log_prob /= float(seq_length)
                    lm_log_prob /= float(seq_length)
                # made_logits = cands[torch.arange(0, bs), made_slt,:] #.float() * made_loss[torch.arange(0, bs), made_slt].unsqueeze(-1)
                dual_log_prob = nlu_log_prob - self.made_weight*made_log_prob.cuda() + self.lm_weight*lm_log_prob
            else:
                dual_log_prob = nlu_log_prob

            nlg_seq_logits = logits.contiguous().view(batch_size*sample_size, seq_length, vocab_size)
            nlg_log_prob = - torch.nn.functional.cross_entropy(nlg_seq_logits.contiguous().view(-1, vocab_size), nlg_outputs.contiguous().view(-1), ignore_index=_PAD, reduction='none')
            nlg_log_prob = nlg_log_prob.view(batch_size, sample_size, seq_length).sum(-1)
            if norm_seq_length:
                nlg_log_prob /= float(seq_length)

            nlg_joint_prob = (1 - self.nlg_weight) * dual_log_prob + self.nlg_weight *nlg_log_prob
            # nlu_logits = nlu_model(nlu_cands) # [BSZxBEAM, ATTR]


            # weight scores between nlu and nlg
            cands_highest = nlg_joint_prob.argmax(-1)
            best = nlg_decisions[torch.arange(0, len(cands_highest)), cands_highest].detach()
        else:
            best = nlg_decisions[:, 0].detach()
        return best
