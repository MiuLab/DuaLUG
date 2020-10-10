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
# from utils import single_BLEU, BLEU, single_ROUGE, ROUGE, best_ROUGE, print_time_info, check_dir, print_curriculum_status
from utils import *
from text_token import _UNK, _PAD, _BOS, _EOS
from model_utils import collate_fn_nlg, collate_fn_nlu, collate_fn_nl, collate_fn_sf, build_optimizer, get_device
from logger import Logger
#from data_engine import DataEngineSplit

from tqdm import tqdm
import pdb


from module_marginal import MaskPredict



class Marginal:
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
            dim_loss=False,
            with_intent=True,
            nlg_path=None
        ):

        # Initialize attributes
        # model_dir = os.path.join(model_dir, dir_name)

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        self.model_dir = model_dir
        self.log_dir = log_dir
        self.dir_name = dir_name

        self.device = get_device(device)

        self.maskpredict = MaskPredict(
                dim_embedding=dim_embedding,
                dim_hidden=dim_hidden,
                # attr_vocab_size=attr_vocab_size,
                vocab_size=train_data_engine.tokenizer.get_vocab_size(),
                # n_slot_key=len(train_data_engine.slot_vocab),
                n_slot_key=len(train_data_engine.nlg_slot_vocab),
                n_intent=len(train_data_engine.intent_vocab),
                n_layers=n_layers,
                bidirectional=False,
                batch_size=batch_size)

        self.optimizer = torch.optim.Adam(
            self.maskpredict.parameters(), 3e-4, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.1
        )

        if is_load:
            print_time_info("Loading marginal model from %s"%self.model_dir)
            self.load_model(self.model_dir)
        else:
            pass
            # self.nlg = NLGRNN(
            #     dim_embedding=dim_embedding,
            #     dim_hidden=dim_hidden,
            #     # attr_vocab_size=attr_vocab_size,
            #     vocab_size=train_data_engine.tokenizer.get_vocab_size(),
            #     # n_slot_key=len(train_data_engine.slot_vocab),
            #     n_slot_key=len(train_data_engine.nlg_slot_vocab),
            #     n_intent=len(train_data_engine.intent_vocab),
            #     n_layers=n_layers,
            #     bidirectional=False,
            #     batch_size=batch_size)
            # pretrained_nlg = torch.load(nlg_path)
            # self.maskpredict.load_encoder(pretrained_nlg)

        self.train_data_engine = train_data_engine
        self.test_data_engine = test_data_engine

        self.train_nlg_data_loader = DataLoader(
            train_data_engine,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
            collate_fn=train_data_engine.collate_fn_nlg,
            pin_memory=True)

        self.test_nlg_data_loader = DataLoader(
            test_data_engine,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
            collate_fn=test_data_engine.collate_fn_nlg,
            pin_memory=True)

        self.maskpredict_parameters = filter(
                lambda p: p.requires_grad, self.maskpredict.parameters())
        self.maskpredict_optimizer = build_optimizer(
                optimizer, self.maskpredict_parameters,
                learning_rate)

        self.train_log_path = os.path.join(self.log_dir, "train_log.csv")
        self.valid_log_path = os.path.join(self.log_dir, "valid_log.csv")

        self.test_result_path = os.path.join(self.log_dir, "test_result.txt")

        with open(self.train_log_path, 'w') as file:
            file.write("epoch,loss\n")
        with open(self.valid_log_path, 'w') as file:
            file.write("epoch,loss\n")

    def get_log_prob(self, inputs, n_samples=1):
        """
        args:
            inputs: tensor, shape [batch_size, attr_vocab_size]

        returns:
            log_probs: tensor, shape [batch_size]
                       log-probability of inputs
        """
        with torch.no_grad():
            log_prob = self.maskpredict.get_log_prob(inputs, n_samples)
        return log_prob.detach().cpu()

        # probs = logits.sigmoid().cpu()
        # return (probs * inputs + (1-probs) * (1-inputs)).log().sum(-1)

    def train(self, epochs, batch_size):
        for idx in range(epochs):
            self.scheduler.step(idx)
            epoch_loss = 0
            batch_amount = 0
            max_norm = 5.0
            # np.random.shuffle(data)
            # steps = len(data) // batch_size
            pbar = tqdm(self.train_nlg_data_loader)
            for step, batch in enumerate(pbar):
                attrs = (
                    batch['slot_key'].clone().detach().to(self.device),
                    torch.tensor(batch['slot_key_lens']).to(self.device),
                    batch['slot_value'].clone().detach().to(self.device),
                    torch.tensor(batch['slot_value_lens']).to(self.device),
                    batch['intent'].clone().detach().to(self.device))
                # batch = data[step*batch_size:(step+1)*batch_size]
                batch_loss = self.maskpredict(attrs)
                # print(self.maskpredict.get_log_prob(attrs).mean())
                epoch_loss += float(batch_loss.item())
                batch_amount += 1
                batch_loss.backward()
                # if max_norm:
                #     clip_grad_norm_(self.maskpredict_parameters, max_norm)
                self.maskpredict_optimizer.step()
                self.maskpredict_optimizer.zero_grad()
                pbar.set_postfix(Loss="{:.5f}".format(epoch_loss / batch_amount))

            epoch_loss /= batch_amount
            print_time_info("Epoch {} finished, training loss {}".format(
                    idx, epoch_loss))
            self.test(batch_size)
            print_time_info("Epoch {}: save model...".format(idx))
            self.save_model(self.model_dir)

    def test(self, batch_size, n_samples=5):
        with torch.no_grad():
            test_loss = 0
            batch_amount = 0
            # steps = len(data) // batch_size
            pbar = tqdm(self.test_nlg_data_loader)
            for step, batch in enumerate(pbar):
                attrs = (
                    batch['slot_key'].clone().detach().to(self.device),
                    torch.tensor(batch['slot_key_lens']).to(self.device),
                    batch['slot_value'].clone().detach().to(self.device),
                    torch.tensor(batch['slot_value_lens']).to(self.device),
                    batch['intent'].clone().detach().to(self.device))
                batch_loss = self.maskpredict(attrs)
                test_loss += batch_loss.item()
                batch_amount += 1

            test_loss /= batch_amount
            print_time_info("testing finished, testing loss {}".format(test_loss))

    def save_model(self, model_dir):
        path = os.path.join(model_dir, "maskpredict.ckpt")
        torch.save(self.maskpredict, path)
        print_time_info("Save model successfully")

    def load_model(self, model_dir):
        path = os.path.join(model_dir, "maskpredict.ckpt")
        if not os.path.exists(path):
            print_time_info("Loading failed, start training from scratch...")
        else:
            self.maskpredict = torch.load(path, map_location=self.device)
            print_time_info("Load model from {} successfully".format(model_dir))

    @classmethod
    def load_pretrained(cls,
                        model_dir,
                        device=None):
        config_path = os.path.join(model_dir, "marginal_config")
        args = dict()
        for line in open(config_path):
            name, value = line.strip().split(': ', maxsplit=1)
            args[name] = value

        model = cls(**args)
        path = os.path.join(model_dir, "maskpredict.ckpt")
        model = torch.load(path)
        return model
