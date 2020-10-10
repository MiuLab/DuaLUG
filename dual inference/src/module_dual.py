import random
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pyemd import emd_samples
from sklearn.metrics import f1_score
from utils import *
from attn import ScaleDotAttention
import pdb

class Criterion(nn.Module):
    def __init__(self, model, reward_type, loss_weight,
                 supervised=True, rl_lambda=1.0, rl_alpha=0.5,
                 pretrain_epochs=0, total_epochs=-1, anneal_type='none',
                 LM=None, MADE=None, training_set_label_samples=None):
        super(Criterion, self).__init__()
        self.model = model
        self.reward_type = reward_type
        self.supervised = supervised
        self.rl_lambda = rl_lambda
        self.rl_alpha = rl_alpha
        self.pretrain_epochs = pretrain_epochs
        self.epoch = 0
        self.total_epochs = total_epochs
        self.anneal_type = anneal_type
        if anneal_type == 'linear' and (total_epochs is None):
            raise ValueError("Please set total_epochs if you want to " \
                             "use anneal_type='linear'")
        if anneal_type == 'switch' and pretrain_epochs == 0:
            raise ValueError("Please set pretrain_epochs > 0 if you want to " \
                             "use anneal_type='switch'")
        self.LM = LM
        self.MADE = MADE
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        # self.CE = nn.CrossEntropyLoss(weight=loss_weight, reduction='none')
        self.CE = nn.CrossEntropyLoss(weight=loss_weight, reduction='none', ignore_index=-1)
        if 'em' in reward_type:
            samples = sum(training_set_label_samples, [])
            np.random.shuffle(samples)
            n = 10
            size = len(samples) // n
            self.samples = [
                samples[i*size:(i+1)*size]
                for i in range(n)
            ]

    def set_scorer(self, scorer):
        self.scorer = scorer

    def epoch_end(self):
        self.epoch += 1
        if self.anneal_type != 'none' and self.epoch == self.pretrain_epochs:
            print_time_info("loss scheduling started ({})".format(self.anneal_type))

    def earth_mover(self, decisions):
        # decisions.size() == (batch_size, sample_size, attr_vocab_size)
        length = decisions.size(-1)
        indexes = (decisions.float().numpy() >= 0.5)
        emd = [
            [
                emd_samples(
                    np.arange(length)[index].tolist(),
                    self.samples[0]
                ) if index.sum() > 0 else 1.0
                for index in indexes[bid]
            ]
            for bid in range(decisions.size(0))
        ]
        return torch.tensor(emd, dtype=torch.float, device=decisions.device)

    def get_scheduled_loss(self, sup_loss, rl_loss):
        if self.epoch < self.pretrain_epochs:
            return sup_loss, 0
        elif self.anneal_type == 'none':
            return sup_loss, rl_loss
        elif self.anneal_type == 'switch':
            return 0, rl_loss

        assert self.anneal_type == 'linear'
        rl_weight = (self.epoch - self.pretrain_epochs + 1) / (self.total_epochs - self.pretrain_epochs + 1)
        return (1-rl_weight) * sup_loss, rl_weight * rl_loss

    def get_scores(self, name, logits):
        size = logits.size(0)
        ret = torch.tensor(getattr(self.scorer, name)[-size:]).float()
        if len(ret.size()) == 2:
            ret = ret.mean(dim=-1)
        return ret

    def get_log_joint_prob_nlg(self, logits, decisions):
        """
        args:
            logits: tensor of shape [batch_size, beam_size, seq_length, vocab_size]
            decisions: tensor of shape [batch_size, beam_size, seq_length, vocab_size]
                       one-hot vector of decoded word-ids
        returns:
            log_joint_prob: tensor of shape [batch_size, beam_size]
        """
        logits = logits.contiguous().view(*decisions.size())
        probs = torch.softmax(logits, dim=-1)
        return (decisions * probs).sum(dim=-1).log().sum(dim=-1)

    def get_log_joint_prob_nlu(self, logits, decisions):
        """
        args:
            logits: tensor of shape [batch_size, attr_vocab_size]
                    or [batch_size, sample_size, attr_vocab_size]
            decisions: tensor of shape [batch_size, sample_size, attr_vocab_size]
                       decisions(0/1)
        returns:
            log_joint_prob: tensor of shape [batch_size, sample_size]
        """
        if len(logits.size()) == len(decisions.size()) - 1:
            logits = logits.unsqueeze(1).expand(-1, decisions.size(1), -1)

        probs = torch.sigmoid(logits)
        decisions = decisions.float()
        probs = probs * decisions + (1-probs) * (1-decisions)
        return probs.log().sum(dim=-1)

    def lm_log_prob(self, decisions):
        # decisions.size() == (batch_size, beam_size, seq_length, vocab_size)
        log_probs = [
            self.LM.get_log_prob(decisions[:, i])
            for i in range(decisions.size(1))
        ]
        return torch.stack(log_probs, dim=0).transpose(0, 1)

    def made_log_prob(self, decisions):
        log_probs = [
            self.MADE.get_log_prob(decisions[:, i].float())
            for i in range(decisions.size(1))
        ]
        return torch.stack(log_probs, dim=0).transpose(0, 1)

    def nlg_loss(self, logits, targets):
        bs = targets.size(0)
        loss = [
            self.CE(logits[:, i].contiguous().view(-1, logits.size(-1)), targets.view(-1)).view(bs, -1).mean(-1)
            for i in range(logits.size(1))
        ]
        return torch.stack(loss, dim=0).transpose(0, 1)

    def nlg_score(self, decisions, targets, func):
        scores = [
            func(targets, np.argmax(decisions.detach().cpu().numpy()[:, i], axis=-1))
            for i in range(decisions.size(1))
        ]
        scores = torch.tensor(scores, dtype=torch.float, device=decisions.device).transpose(0, 1)
        if len(scores.size()) == 3:
            scores = scores.mean(-1)

        return scores

    def nlu_loss(self, logits, targets):
        loss = [
            self.BCE(logits[:, i], targets).mean(-1)
            for i in range(logits.size(1))
        ]
        return torch.stack(loss, dim=0).transpose(0, 1)

    def nlu_score(self, decisions, targets, average):
        device = decisions.device
        decisions = decisions.detach().cpu().long().numpy()
        targets = targets.detach().cpu().long().numpy()
        scores = [
            [
                f1_score(y_true=np.array([label]), y_pred=np.array([pred]), average=average)
                for label, pred in zip(targets, decisions[:, i])
            ]
            for i in range(decisions.shape[1])
        ]
        return torch.tensor(scores, dtype=torch.float, device=device).transpose(0, 1)

    def get_reward(self, logits, targets, decisions=None):
        reward = 0
        if decisions is not None:
            decisions = decisions.detach()

        if self.model == "nlu":
            if self.reward_type == "loss":
                reward = self.nlu_loss(logits, targets)
            elif self.reward_type == "micro-f1":
                reward = -self.nlu_score(decisions, targets, 'micro')
            elif self.reward_type == "weighted-f1":
                reward = -self.nlu_score(decisions, targets, 'weighted')
            elif self.reward_type == "f1":
                reward = -(self.nlu_score(decisions, targets, 'micro') + self.nlu_score(decisions, targets, 'weighted'))
            elif self.reward_type == "em":
                reward = self.earth_mover(decisions)
            elif self.reward_type == "made":
                reward = -self.made_log_prob(decisions)
            elif self.reward_type == "loss-em":
                reward = self.nlu_loss(logits, targets) + self.earth_mover(decisions)
        elif self.model == "nlg":
            if self.reward_type == "loss":
                reward = self.nlg_loss(logits, targets)
            elif self.reward_type == "lm":
                reward = -self.lm_log_prob(decisions)
            elif self.reward_type == "bleu":
                reward = -self.nlg_score(decisions, targets, func=single_BLEU)
            elif self.reward_type == "rouge":
                reward = -self.nlg_score(decisions, targets, func=single_ROUGE)
            elif self.reward_type == "bleu-rouge":
                reward = -(self.nlg_score(decisions, targets, func=single_BLEU) + self.nlg_score(decisions, targets, func=single_ROUGE))
            elif self.reward_type == "loss-lm":
                reward = self.nlg_loss(logits, targets) - self.lm_log_prob(decisions)

        return reward

    def forward(self, logits, targets, decisions=None, \
            intent_logits=None, intent_targets=None, \
            n_supervise=1, log_joint_prob=None, supervised=True, last_reward=0.0, calculate_reward=True):
        """
        args:
            logits: tensor of shape [batch_size, sample_size, * ]
            targets: tensor of shape [batch_size, *]
            decisions: tensor of shape [batch_size, sample_size, *]
        """
        if not self.supervised:
            supervised = False

        logits = logits.contiguous()
        targets = targets.contiguous()

        sup_loss = rl_loss = 0
        reward = 0.0
        if self.epoch >= self.pretrain_epochs and calculate_reward:
            reward = self.rl_lambda * self.get_reward(logits, targets, decisions)
        if isinstance(last_reward, torch.Tensor):
            reward = self.rl_alpha * last_reward + (1 - self.rl_alpha) * reward
        
        if self.model == "nlu":
            if supervised:
                splits = logits.split(split_size=1, dim=1)
                # if intent_logits:
                #     intent_splits = intent_logits.split(split_size=1, dim=1)
                intent_splits = intent_logits.split(split_size=1, dim=1)
                for i in range(n_supervise):
                    sup_loss += self.CE(splits[i].contiguous().view(-1, logits.size(-1)), targets.view(-1)).mean()
                    # if intent_logits:
                    #     # intent to one-hot
                    #     intent_targets = torch.zeros_like(intent_splits[i].squeeze(1)).scatter_(1, intent_targets.unsqueeze(1), 1)
                    #     sup_loss += self.BCE(intent_splits[i].squeeze(1), intent_targets).mean()
                    intent_targets = torch.zeros_like(intent_splits[i].squeeze(1)).scatter_(1, intent_targets.unsqueeze(1), 1)
                    sup_loss += self.BCE(intent_splits[i].squeeze(1), intent_targets).mean()
            # X = self.get_log_joint_prob_nlu(logits, decisions) if log_joint_prob is None else log_joint_prob
            X = self.get_log_joint_prob_nlg(logits, decisions) if log_joint_prob is None else log_joint_prob
        elif self.model == "nlg":
            if supervised:
                splits = logits.split(split_size=1, dim=1)
                for i in range(n_supervise):
                    sup_loss += self.CE(splits[i].contiguous().view(-1, logits.size(-1)), targets.view(-1)).mean()
            X = self.get_log_joint_prob_nlg(logits, decisions) if log_joint_prob is None else log_joint_prob
        
        if isinstance(reward, torch.Tensor):
            rl_loss = (reward * X).mean()

        sup_loss, rl_loss = self.get_scheduled_loss(sup_loss, rl_loss)

        return sup_loss, rl_loss, X, reward


class RNNModel(nn.Module):
    def __init__(self,
                 dim_embedding,
                 dim_hidden,
                 vocab_size,
                 n_layers=1,
                 bidirectional=False):
        super(RNNModel, self).__init__()
        self.dim_embedding = dim_embedding
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        self.rnn = nn.GRU(dim_embedding,
                          dim_hidden,
                          num_layers=n_layers,
                          batch_first=True,
                          bidirectional=bidirectional)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _init_hidden(self, inputs):
        """
        args:
            inputs: shape [batch_size, *]
                    a input tensor with correct device

        returns:
            hidden: shpae [n_layers*n_directions, batch_size, dim_hidden]
                    all-zero hidden state
        """
        batch_size = inputs.size(0)
        return torch.zeros(self.n_layers*self.n_directions,
                           batch_size,
                           self.dim_hidden,
                           dtype=torch.float,
                           device=inputs.device)

def segment_and_padding(batch, lens):
    '''
    batch: shape [batch_size x dynamic_slot_size, emb_size]
    lens: list [batch_size]
    '''
    ret = []
    max_len = max(lens)
    pos = 0
    for i, l in enumerate(lens):
        padding = torch.zeros(max_len - l, batch.shape[-1]).type_as(batch).to(batch.device)
        padded = torch.cat([batch[pos:pos + l], padding], dim=0)
        ret.append(padded)
        pos += l
    assert pos == sum(lens)
    return torch.stack(ret, dim=0)

def pass_rnn(rnn, inputs, hidden, lens):
    packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, lens, batch_first=True, enforce_sorted=False)
    output, _ = rnn(packed, hidden)
    output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
    return output

class NLGRNN(RNNModel):
    def __init__(self,
                 dim_embedding,
                 dim_hidden,
                 vocab_size,
                 n_slot_key,
                 n_intent,
                 n_layers=1,
                 bidirectional=True,
                 batch_size=16):
        super(NLGRNN, self).__init__(dim_embedding,
                                     dim_hidden,
                                     vocab_size,
                                     n_layers=1,
                                     bidirectional=False)
        if self.n_directions != 1:
            raise ValueError("RNN must be uni-directional in NLG model.")

        self.batch_size = batch_size
        self.slot_key_embedding = nn.Embedding(n_slot_key, dim_embedding)
        self.intent_embedding = nn.Embedding(n_intent, dim_hidden)
        self.slot_type_encoder = nn.GRU(dim_embedding,
                    dim_hidden,
                    num_layers=n_layers,
                    batch_first=True,
                    bidirectional=True)
        self.slot_type_fc = nn.Linear(dim_hidden*2, dim_hidden)
        self.slot_encoder = nn.GRU(dim_embedding,
                    dim_hidden,
                    num_layers=n_layers,
                    batch_first=True,
                    bidirectional=True)
        self.slot_fc = nn.Linear(dim_hidden*2, dim_hidden)
        self.global_encoder = nn.GRU(dim_hidden,
                    dim_hidden,
                    num_layers=n_layers,
                    batch_first=True,
                    bidirectional=True)
        self.global_fc = nn.Linear(dim_hidden*2, dim_hidden)
        self.linear_q = nn.Linear(dim_embedding+self.dim_hidden, self.dim_hidden)
        self.linear_k = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.linear_v = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.attention = ScaleDotAttention(2.0, 1)
        self.attn_merge = nn.Linear(self.dim_hidden + self.dim_embedding, self.dim_embedding)
        self.hidden_merge = nn.Linear(self.dim_hidden * 2, self.dim_hidden)
        self.linear = nn.Linear(self.dim_hidden, self.vocab_size)


    def encode_slot_intent(self, slot_key, slot_key_lens, slot_value, slot_value_lens, intent, hidden=None):
        '''
        Arguments: 
        slot_key: shape [batch_size x dynamic_slot_size]
        slot_key_lens: list [batch_size]
        slot_value: shape [batch_size x dynamic_slot_size, max_value_len]
        slot_value_lens: list [batch_size x dynamic_slot_size]
        intent: shape [batch_size]
        
        '''
        emb_slot_key = self.slot_key_embedding(slot_key)
        # compute init hidden state
        slot_type_emb = segment_and_padding(emb_slot_key, slot_key_lens)
        assert 0 not in slot_key_lens
        try:
            # slot_key_lens = torch.clamp(slot_key_lens, 1, 10000)
            slot_type_encoded = self.slot_type_fc(pass_rnn(self.slot_type_encoder, slot_type_emb, None, slot_key_lens))
        except:
            pdb.set_trace()
        intent_feats = self.intent_embedding(intent)
        self.cache_h = self.hidden_merge(torch.cat([slot_type_encoded.mean(dim=1), intent_feats], dim=-1))
        # compute slot representation
        emb_slot_key = emb_slot_key.unsqueeze(1)
        emb_slot_value = self.embedding(slot_value)
        emb_slot = torch.cat([emb_slot_key, emb_slot_value], dim = 1)
        slot_lens = [(x + 1) for x in slot_value_lens]
        encoded_slot = self.slot_fc(pass_rnn(self.slot_encoder, emb_slot, hidden, slot_lens))
        # packed = torch.nn.utils.rnn.pack_padded_sequence(emb_slot, slot_lens, batch_first=True, enforce_sorted=False)
        # encoded_slot, _ = self.slot_encoder(packed, hidden)
        # encoded_slot, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded_slot, batch_first = True)
        for i, value_len in enumerate(slot_lens):
            encoded_slot[i, value_len:,:] = 0.0
        encoded_slot_sum = torch.sum(encoded_slot, dim=1)
        slot_lens_tensor = torch.tensor(slot_lens).to(encoded_slot_sum.device).type_as(encoded_slot_sum).unsqueeze(-1)
        encoded_slot_mean = encoded_slot_sum / slot_lens_tensor
        global_feats = segment_and_padding(encoded_slot_mean, slot_key_lens)
        encoded_global_feats = self.global_fc(pass_rnn(self.global_encoder, global_feats, None, slot_key_lens))
        final_feats = torch.cat([intent_feats.unsqueeze(1), encoded_global_feats], dim=1)
        self.cache_k = self.linear_k(final_feats)
        self.cache_v = self.linear_v(final_feats)
        self.cache_k_lens = [(x + 1) for x in slot_key_lens]
        self.attention.compute_mask(self.cache_k, self.cache_k_lens)
        
        return self.cache_h

    def clean_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.cache_h = None
        self.cache_k_lens = None
    
    def scale_cache(self, beam_size):
        self.cache_k = self.cache_k.unsqueeze(0).repeat(beam_size, 1,1,1).view(-1, self.cache_k.shape[1], self.cache_k.shape[2])
        self.cache_v = self.cache_v.unsqueeze(0).repeat(beam_size, 1,1,1).view(-1, self.cache_v.shape[1], self.cache_v.shape[2])
        if type(self.cache_k_lens) == list:
            self.cache_k_lens = torch.tensor(self.cache_k_lens, device=self.cache_v.device)
        self.cache_k_lens = self.cache_k_lens.unsqueeze(0).repeat(beam_size, 1).view(-1)
        self.attention.compute_mask(self.cache_k, self.cache_k_lens)

    def get_attn_vectors(self, curr_inputs, cur_hiddens):
        '''
        curr_inputs: shape [batch_size, 1, embedding_dim]
        '''
        query = self.linear_q(torch.cat([curr_inputs, cur_hiddens], dim=-1))
        context, attn = self.attention(query, self.cache_k, self.cache_v)
        return context


    def _st_softmax(self, logits, hard=False, dim=-1):
        y_soft = logits.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret

    def _st_onehot(self, logits, indices, hard=True, dim=-1):
        y_soft = logits.softmax(dim)
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long().to(logits.device)
        if len(logits.size()) == len(indices.size()) + 1:
            indices = indices.unsqueeze(-1)
        y_hard = torch.zeros_like(logits).scatter_(dim, indices, 1.0)
        if hard:
            return y_hard - y_soft.detach() + y_soft, y_hard
        else:
            return y_soft, y_hard

    def forward(self, attrs, bos_id, labels=None,
                tf_ratio=0.5, max_decode_length=50, beam_size=5, st=True):
        """
        args:
            attrs: (slot_key, slot_key_lens, slot_value, slot_value_lens, intent)
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, beam_size, seq_length, vocab_size]
            outputs: shape [batch_size, beam_size, seq_length, vocab_size]
                     output words as one-hot vectors (maybe soft)
            decisions: shape [batch_size, beam_size, seq_length, vocab_size]
                       output words as one-hot vectors (hard)
        """
        if beam_size == 1:
            logits, outputs, semantic_embs = self.forward_greedy(
                attrs, bos_id, labels,
                tf_ratio=tf_ratio, max_decode_length=max_decode_length,
                st=st
            )
            return logits.unsqueeze(1), outputs.unsqueeze(1), outputs.unsqueeze(1), semantic_embs

        decode_length = max_decode_length if labels is None else labels.size(1)

        batch_size = attrs[-1].size(0)
        # hidden.size() should be (n_layers*n_directions, beam_size*batch_size, dim_hidden)
        # semantic_embs = self.transform(attrs.float()).unsqueeze(0).unsqueeze(0)
        # semantic_embs = self.transform(attrs.float())
        # pdb.set_trace()
        self.encode_slot_intent(*attrs)
        self.scale_cache(beam_size)
        semantic_embs = self.cache_h
        hiddens = self.cache_h.unsqueeze(0).unsqueeze(0)
        hiddens = hiddens.expand(self.n_layers*self.n_directions, beam_size, -1, -1)
        hiddens = hiddens.contiguous().view(-1, beam_size*batch_size, self.dim_hidden)
        last_output = torch.full([batch_size], bos_id, dtype=torch.long)
        # last_output.size() == (beam_size, batch_size)
        last_output = [last_output for _ in range(beam_size)]
        # logits.shape will be [seq_length, beam_size, batch_size, vocab_size]
        logits = []
        beam_probs = np.full((beam_size, batch_size), -math.inf)
        beam_probs[0, :] = 0.0
        # last_indices.shape will be [seq_length, batch_size, beam_size]
        last_indices = []
        output_ids = []
        for step in range(decode_length):
            curr_inputs = []
            for beam in range(beam_size):
                use_tf = False if step == 0 else random.random() < tf_ratio
                if use_tf:
                    curr_input = labels[:, step-1]
                else:
                    curr_input = last_output[beam].cuda().detach()

                if len(curr_input.size()) == 1:
                    # curr_input are ids
                    curr_input = self.embedding(curr_input).unsqueeze(1)
                else:
                    # curr_input are one-hot vectors
                    curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)
                curr_inputs.append(curr_input)

            curr_inputs = torch.stack(curr_inputs, dim=0)
            curr_inputs = curr_inputs.view(-1, 1, self.dim_embedding)
            # curr_inputs.size() == (beam_size, batch_size, 1, dim_embedding)
            # pdb.set_trace()
            attn_vectors = self.get_attn_vectors(curr_inputs.squeeze(), hiddens[-1, ...])
            pre_curr = torch.cat([curr_inputs, attn_vectors.unsqueeze(1)], dim=-1)
            curr_inputs = self.attn_merge(pre_curr)
            # curr_inputs = curr_inputs.view(-1, 1, self.dim_embedding)
            # pdb.set_trace()
            ## add attrs embedding to input
            output, new_hiddens = self.rnn(curr_inputs, hiddens)
            output = self.linear(output.squeeze(1))
            output = output.view(beam_size, batch_size, -1)
            new_hiddens = new_hiddens.view(self.n_layers*self.n_directions, beam_size, batch_size, -1)
            probs = torch.log_softmax(output.detach(), dim=-1)
            # top_probs.size() == top_indices.size() == (beam_size, batch_size, k)
            top_probs, top_indices = torch.topk(probs, k=beam_size, dim=-1)
            top_probs = top_probs.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()
            last_index = []
            output_id = []
            for bid in range(batch_size):
                beam_prob = []
                for beam in range(beam_size):
                    beam_prob.extend([
                        (
                            beam,
                            top_indices[beam, bid, i],
                            beam_probs[beam][bid] + top_probs[beam, bid, i]
                        )
                        for i in range(beam_size)
                    ])
                topk = sorted(beam_prob, key=lambda x: x[2], reverse=True)[:beam_size]
                last_index.append([item[0] for item in topk])
                output_id.append([item[1] for item in topk])
                beam_probs[:, bid] = np.array([item[2] for item in topk])

            last_indices.append(last_index)
            output_ids.append(output_id)

            new_hiddens = new_hiddens.permute([2, 0, 1, 3]).split(split_size=1, dim=0)
            hiddens = torch.stack([
                new_hiddens[bid].squeeze(0).index_select(dim=1, index=torch.tensor(indices).to(new_hiddens[bid].device))
                for bid, indices in enumerate(last_index)
            ], dim=0).permute([1, 2, 0, 3]).contiguous().view(-1, beam_size*batch_size, self.dim_hidden)

            output = output.transpose(0, 1).split(split_size=1, dim=0)
            output = [
                output[bid].squeeze(0).index_select(dim=0, index=torch.tensor(indices).to(output[bid].device))
                for bid, indices in enumerate(last_index)
            ]
            logits.append(output)

            last_output = [
                torch.tensor(
                    [output_id[bid][beam] for bid in range(batch_size)],
                    dtype=torch.long, device=attrs[0].device
                )
                for beam in range(beam_size)
            ]

        last_indices = np.array(last_indices)
        output_ids = np.array(output_ids)
        # back-trace the beams to get outputs
        beam_outputs = []
        beam_logits = []
        beam_decisions = []
        for bid in range(batch_size):
            this_index = np.arange(beam_size)
            step_logits = []
            step_output_ids = []
            for step in range(decode_length-1, -1, -1):
                this_logits = logits[step][bid].index_select(dim=0, index=torch.from_numpy(this_index).to(logits[step][bid].device))
                step_logits.append(this_logits)
                step_output_ids.append(output_ids[step, bid, this_index])
                this_index = last_indices[step, bid, this_index]

            step_logits = torch.stack(step_logits[::-1], dim=0)
            step_outputs, step_decisions = self._st_onehot(step_logits, np.array(step_output_ids[::-1]), hard=st)
            beam_outputs.append(step_outputs)
            beam_logits.append(step_logits)
            beam_decisions.append(step_decisions)

        logits = torch.stack(beam_logits).transpose(1, 2)
        outputs = torch.stack(beam_outputs).transpose(1, 2)
        decisions = torch.stack(beam_decisions).transpose(1, 2)
        return logits, outputs, decisions, semantic_embs

    def forward_greedy(self, attrs, bos_id, labels=None, sampling=False,
                       tf_ratio=0.5, max_decode_length=50, st=True):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size]
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, seq_length, vocab_size]
            outputs: shape [batch_size, seq_length, vocab_size]
                     output words as one-hot vectors
        """
        batch_size = attrs[-1].size(0)
        # batch_size = self.batch_size
        decode_length = max_decode_length if labels is None else labels.size(1)
        self.encode_slot_intent(*attrs)
        hidden = self.cache_h.unsqueeze(0)
        semantic_embs = hidden
        # hidden = self.transform(attrs.float()).unsqueeze(0)
        # semantic_embs = hidden.transpose(0, 1)
        # pdb.set_trace()
        hidden = hidden.expand(self.n_layers*self.n_directions, -1, -1).contiguous()
        # last_output = torch.full([batch_size], bos_id, dtype=torch.long)
        # why?
        last_output = torch.full([batch_size], bos_id, dtype=torch.long).to(attrs[0].device)
        
        logits = []
        outputs = []
        for step in range(decode_length):
            use_tf = False if step == 0 else random.random() < tf_ratio
            if use_tf:
                curr_input = labels[:, step-1]
            else:
                curr_input = last_output.detach()

            if len(curr_input.size()) == 1:
                curr_input = self.embedding(curr_input).unsqueeze(1)
            else:
                curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)

            # if self.cat_semantic:
            #     pre_curr = torch.cat([curr_input, semantic_embs], dim=-1)
            #     curr_input = self.merge(pre_curr)
            # pdb.set_trace()
            try:
                attn_vectors = self.get_attn_vectors(curr_input.squeeze(), hidden[-1, ...])
            except:
                pdb.set_trace()
            pre_curr = torch.cat([curr_input, attn_vectors.unsqueeze(1)], dim=-1)
            curr_input = self.attn_merge(pre_curr)
            output, hidden = self.rnn(curr_input, hidden)
            output = self.linear(output.squeeze(1))
            logits.append(output)
            if sampling:
                last_output = F.gumbel_softmax(output, hard=True)
            else:
                last_output = self._st_softmax(output, hard=True, dim=-1)
            outputs.append(self._st_softmax(output, hard=st, dim=-1))

        logits = torch.stack(logits).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1)
        return logits, outputs, semantic_embs


class NLURNN(RNNModel):
    def __init__(self, 
                 dim_embedding,
                 dim_hidden,
                 vocab_size,
                 slot_vocab_size,
                 intent_vocab_size,
                 n_layers=1,
                 bidirectional=True):
        super(NLURNN, self).__init__(dim_embedding,
                                     dim_hidden,
                                     vocab_size,
                                     n_layers=1,
                                     bidirectional=False)
        self.slot_vocab_size = slot_vocab_size
        self.intent_vocab_size = intent_vocab_size
        self.slot_linear = nn.Linear(
            self.n_layers * self.n_directions * self.dim_hidden,
            self.slot_vocab_size
        )
        self.intent_linear = nn.Linear(
            self.n_layers * self.n_directions * self.dim_hidden,
            self.intent_vocab_size
        )
        self.slot_embedding = nn.Embedding(self.slot_vocab_size, dim_embedding)
        self.input_merge = nn.Linear(2*dim_embedding, dim_embedding)

    def nat_forward(self, inputs, sample_size=1):
        """
        args:
            inputs: shape [batch_size, seq_length]
                    or shape [batch_size, seq_length, attr_vocab_size] one-hot vectors

        outputs:
            logits: shape [batch_size, attr_vocab_size]
        """
        batch_size = inputs.size(0)
        decode_length = inputs.size(1)
        last_output = torch.full([batch_size], 0, dtype=torch.long)
        if len(inputs.size()) == 2:
            inputs = self.embedding(inputs)
        else:
            # suppose the inputs are one-hot vectors
            inputs = torch.matmul(inputs.float(), self.embedding.weight)

        outputs, hidden = self.rnn(inputs)
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        intent_logits = self.intent_linear(hidden)
        slot_logits = self.slot_linear(outputs)
        return slot_logits, intent_logits

    def _st_onehot(self, logits, indices, hard=True, dim=-1):
        y_soft = logits.softmax(dim)
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long().to(logits.device)
        if len(logits.size()) == len(indices.size()) + 1:
            indices = indices.unsqueeze(-1)
        y_hard = torch.zeros_like(logits).scatter_(dim, indices, 1.0)
        if hard:
            return y_hard - y_soft.detach() + y_soft, y_hard
        else:
            return y_soft, y_hard

    def forward(self, inputs, bos_id=0, labels=None,
                tf_ratio=0.5, max_decode_length=50, beam_size=5, st=True, eos_id=3):
        """
        args:
            attrs: (slot_key, slot_key_lens, slot_value, slot_value_lens, intent)
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, beam_size, seq_length, vocab_size]
            outputs: shape [batch_size, beam_size, seq_length, vocab_size]
                     output words as one-hot vectors (maybe soft)
            decisions: shape [batch_size, beam_size, seq_length, vocab_size]
                       output words as one-hot vectors (hard)
        """
        batch_size = inputs.size(0)
        decode_length = inputs.size(1)
        input_lens = []
        for i, data in enumerate(inputs):
            if eos_id in data:
                input_lens.append(data.tolist().index(eos_id) + 1)
            else:
                input_lens.append(len(data))

        if beam_size == 1:
            logits, outputs, intent_logits = self.forward_greedy(
                inputs, bos_id, labels,
                tf_ratio=tf_ratio, max_decode_length=max_decode_length,
                st=st
            )
            return logits.unsqueeze(1), outputs.unsqueeze(1), outputs.unsqueeze(1), intent_logits

        decode_length = inputs.size(1)

        # batch_size = attrs[-1].size(0)
        # hidden.size() should be (n_layers*n_directions, beam_size*batch_size, dim_hidden)
        # semantic_embs = self.transform(attrs.float()).unsqueeze(0).unsqueeze(0)
        # semantic_embs = self.transform(attrs.float())
        # self.encode_slot_intent(*attrs)
        # self.scale_cache(beam_size)
        # hiddens = self.cache_h.unsqueeze(0).unsqueeze(0)
        # hiddens = hiddens.expand(self.n_layers*self.n_directions, beam_size, -1, -1)
        # hiddens = hiddens.contiguous().view(-1, beam_size*batch_size, self.dim_hidden)
        hiddens = None
        intent_pre_out = torch.zeros(batch_size, self.dim_hidden).to(inputs.device)
        last_output = torch.full([batch_size], bos_id, dtype=torch.long).to(inputs.device)
        # last_output.size() == (beam_size, batch_size)
        last_output = [last_output for _ in range(beam_size)]
        # logits.shape will be [seq_length, beam_size, batch_size, vocab_size]
        logits = []
        beam_probs = np.full((beam_size, batch_size), -math.inf)
        beam_probs[0, :] = 0.0
        # last_indices.shape will be [seq_length, batch_size, beam_size]
        last_indices = []
        output_ids = []
        for step in range(decode_length):
            curr_inputs = []
            last_inputs = []
            for beam in range(beam_size):
                use_tf = False if step == 0 else random.random() < tf_ratio
                # if use_tf:
                curr_input = inputs[:, step]
                # else:
                if use_tf:
                    # black magic here
                    last_input = labels[:, step - 1].clone()
                    last_input += (last_input==-1).type(torch.long) # prevent -1
                else:
                    last_input = last_output[beam].detach()

                if len(curr_input.size()) == 1:
                    # curr_input are ids
                    curr_input = self.embedding(curr_input).unsqueeze(1)
                else:
                    # curr_input are one-hot vectors
                    curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)
                
                if len(last_input.size()) == 1:
                    # why though?
                    last_input = last_input.to(inputs.device)
                    
                    # curr_input are ids
                    try:
                        last_input = self.slot_embedding(last_input).unsqueeze(1)
                    except:
                        pdb.set_trace()
                else:
                    # why though?
                    last_input = last_input.to(inputs.device)
                    
                    # curr_input are one-hot vectors
                    last_input = torch.matmul(last_input.float(), self.slot_embedding.weight).unsqueeze(1)
                # last_input = self.slot_embedding(last_input)
                curr_inputs.append(curr_input)
                last_inputs.append(last_input)

            curr_inputs = torch.stack(curr_inputs, dim=0)
            curr_inputs = curr_inputs.view(-1, 1, self.dim_embedding)
            last_inputs = torch.stack(last_inputs, dim=0)
            last_inputs = last_inputs.view(-1, 1, self.dim_embedding)
            # curr_inputs.size() == (beam_size, batch_size, 1, dim_embedding)
            # pdb.set_trace()
            # attn_vectors = self.get_attn_vectors(curr_inputs.squeeze(), hiddens[-1, ...])
            pre_curr = torch.cat([curr_inputs, last_inputs], dim=-1)
            curr_inputs = self.input_merge(pre_curr)
            # curr_inputs = curr_inputs.view(-1, 1, self.dim_embedding)
            # pdb.set_trace()
            ## add attrs embedding to input
            output, new_hiddens = self.rnn(curr_inputs, hiddens)
            for i in range(batch_size):
                if input_lens[i] - 1 == step:
                    intent_pre_out[i] = output[i]
            output = self.slot_linear(output.squeeze(1))
            output = output.view(beam_size, batch_size, -1)
            new_hiddens = new_hiddens.view(self.n_layers*self.n_directions, beam_size, batch_size, -1)
            probs = torch.log_softmax(output.detach(), dim=-1)
            # top_probs.size() == top_indices.size() == (beam_size, batch_size, k)
            top_probs, top_indices = torch.topk(probs, k=beam_size, dim=-1)
            top_probs = top_probs.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()
            last_index = []
            output_id = []
            for bid in range(batch_size):
                beam_prob = []
                for beam in range(beam_size):
                    beam_prob.extend([
                        (
                            beam,
                            top_indices[beam, bid, i],
                            beam_probs[beam][bid] + top_probs[beam, bid, i]
                        )
                        for i in range(beam_size)
                    ])
                topk = sorted(beam_prob, key=lambda x: x[2], reverse=True)[:beam_size]
                last_index.append([item[0] for item in topk])
                output_id.append([item[1] for item in topk])
                beam_probs[:, bid] = np.array([item[2] for item in topk])

            last_indices.append(last_index)
            output_ids.append(output_id)

            new_hiddens = new_hiddens.permute([2, 0, 1, 3]).split(split_size=1, dim=0)
            hiddens = torch.stack([
                new_hiddens[bid].squeeze(0).index_select(dim=1, index=torch.tensor(indices).to(new_hiddens[bid].device))
                for bid, indices in enumerate(last_index)
            ], dim=0).permute([1, 2, 0, 3]).contiguous().view(-1, beam_size*batch_size, self.dim_hidden)

            output = output.transpose(0, 1).split(split_size=1, dim=0)
            output = [
                output[bid].squeeze(0).index_select(dim=0, index=torch.tensor(indices).to(output[bid].device))
                for bid, indices in enumerate(last_index)
            ]
            logits.append(output)

            last_output = [
                torch.tensor(
                    [output_id[bid][beam] for bid in range(batch_size)],
                    dtype=torch.long, device=inputs.device,
                )
                for beam in range(beam_size)
            ]

        last_indices = np.array(last_indices)
        output_ids = np.array(output_ids)
        # back-trace the beams to get outputs
        beam_outputs = []
        beam_logits = []
        beam_decisions = []
        for bid in range(batch_size):
            this_index = np.arange(beam_size)
            step_logits = []
            step_output_ids = []
            for step in range(decode_length-1, -1, -1):
                this_logits = logits[step][bid].index_select(dim=0, index=torch.from_numpy(this_index).to(logits[step][bid].device))
                step_logits.append(this_logits)
                step_output_ids.append(output_ids[step, bid, this_index])
                this_index = last_indices[step, bid, this_index]

            step_logits = torch.stack(step_logits[::-1], dim=0)
            step_outputs, step_decisions = self._st_onehot(step_logits, np.array(step_output_ids[::-1]), hard=st)
            beam_outputs.append(step_outputs)
            beam_logits.append(step_logits)
            beam_decisions.append(step_decisions)

        logits = torch.stack(beam_logits).transpose(1, 2)
        intent_logits = self.intent_linear(intent_pre_out)
        outputs = torch.stack(beam_outputs).transpose(1, 2)
        decisions = torch.stack(beam_decisions).transpose(1, 2)
        return logits, outputs, decisions, intent_logits

    def _st_softmax(self, logits, hard=False, dim=-1):
        y_soft = logits.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret

    def forward_greedy(self, inputs, bos_id=0, labels=None, sampling=False,
                       tf_ratio=0.5, max_decode_length=50, st=True, eos_id=3):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size]
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, seq_length, vocab_size]
            outputs: shape [batch_size, seq_length, vocab_size]
                     output words as one-hot vectors
        """
        input_lens = []
        for i, data in enumerate(inputs):
            if eos_id in data:
                input_lens.append(data.tolist().index(eos_id) + 1)
            else:
                input_lens.append(len(data))
        batch_size = inputs.size(0)
        decode_length = inputs.size(1)
        intent_pre_out = torch.zeros(batch_size, self.dim_hidden).to(inputs.device)
        # self.encode_slot_intent(*attrs)
        hidden = None
        last_output = torch.full([batch_size], bos_id, dtype=torch.long).to(inputs.device)
        logits = []
        outputs = []
        for step in range(decode_length):
            use_tf = False if step == 0 else random.random() < tf_ratio
            # if use_tf:
            curr_input = inputs[:, step]
            # else:
            if use_tf:
                # black magic here
                last_input = labels[:, step - 1].clone()
                last_input += (last_input==-1).type(torch.long) # prevent -1
            else:
                last_input = last_output.detach()
            
            try:
                # last_input = self.slot_embedding(last_input)
                if len(last_input.size()) == 1:
                    # curr_input are ids
                    last_input = self.slot_embedding(last_input).unsqueeze(1)
                else:
                    # curr_input are one-hot vectors
                    last_input = torch.matmul(last_input.float(), self.slot_embedding.weight).unsqueeze(1)
                last_input = last_input.view(-1, 1, self.dim_embedding)
            except:
                pdb.set_trace()

            if len(curr_input.size()) == 1:
                curr_input = self.embedding(curr_input).unsqueeze(1)
            else:
                curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)

            # if self.cat_semantic:
            pre_curr = torch.cat([curr_input, last_input], dim=-1)
            curr_input = self.input_merge(pre_curr)
            # pdb.set_trace()
            # attn_vectors = self.get_attn_vectors(curr_input.squeeze(), hidden[-1, ...])
            # pre_curr = torch.cat([curr_input, attn_vectors.unsqueeze(1)], dim=-1)
            # curr_input = self.attn_merge(pre_curr)
            output, hidden = self.rnn(curr_input, hidden)
            # pdb.set_trace()
            for i in range(batch_size):
                # check empty
                if len(input_lens) and input_lens[i] - 1 == step:
                    intent_pre_out[i] = output[i]
            output = self.slot_linear(output.squeeze(1))
            logits.append(output)
            if sampling:
                last_output = F.gumbel_softmax(output, hard=True)
            else:
                last_output = self._st_softmax(output, hard=True, dim=-1)
            outputs.append(self._st_softmax(output, hard=st, dim=-1))

        logits = torch.stack(logits).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1)
        intent_logits = self.intent_linear(intent_pre_out)
        return logits, outputs, intent_logits


class LMRNN(RNNModel):
    def __init__(self, *args, **kwargs):
        super(LMRNN, self).__init__(*args, **kwargs)
        if self.n_directions != 1:
            raise ValueError("RNN must be uni-directional in LM model.")
        self.linear = nn.Linear(self.dim_hidden, self.vocab_size)

    def forward(self, inputs):
        """
        args:
            inputs: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, seq_length, vocab_size]
        """
        inputs = self.embedding(inputs)
        output, _ = self.rnn(inputs)
        logits = self.linear(output)
        return logits

"""
Borrowed from https://github.com/karpathy/pytorch-made/blob/master/made.py
"""
class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

"""
Borrowed from https://github.com/karpathy/pytorch-made/blob/master/made.py
"""
class MADENet(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ])
        self.net.pop() # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings

        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    ## NLG ##
    from data_engine_nlg import DataEngine
    from torch.utils.data import Dataset, DataLoader
    dataset = DataEngine("../data/snips", "test")
    d = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=DataEngine.collate_fn)
    model = NLGRNN(50, 200, dataset.tokenizer.get_vocab_size(),
            n_slot_key=len(dataset.slot_vocab), n_intent=len(dataset.intent_vocab),
            n_layers=1)
    for b in d:
        break
    model.encode_slot_intent(b['slot_key'], b['slot_key_lens'], b['slot_value'], b['slot_value_lens'], b['intent'])
    
    bos = dataset.tokenizer.token_to_id('[CLS]')
    attrs = (b['slot_key'], b['slot_key_lens'], b['slot_value'], b['slot_value_lens'], b['intent'])
    # logits, outputs, decisions = model.forward(attrs, bos, b['target'])
    # logits, outputs, decisions = model.forward(attrs, bos, b['target'], beam_size=1)
    ## NLU ##
    from data_engine_nlu import DataEngine
    dataset = DataEngine("../data/snips", "test")
    d = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=DataEngine.collate_fn)
    for b in d:
        break
    model = NLURNN(50, 200, dataset.tokenizer.get_vocab_size(),
            slot_vocab_size=len(dataset.slot_vocab), intent_vocab_size=len(dataset.intent_vocab),
            n_layers=1)
    slot_logits, slot_softmax, decisions, intent_logits = model(b['inputs'])