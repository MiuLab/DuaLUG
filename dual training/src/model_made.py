import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import random
import numpy as np
import os
import math

from module import MADENet
from utils import *
from model_utils import get_device

from tqdm import tqdm


class MADE:
    def __init__(self, dim_input, hidden_list, resample_every=20, num_masks=1,
                 model_dir="./model", is_load=True,
                 device=None, dir_name='test'):

        # Initialize attributes
        model_dir = os.path.join(model_dir, dir_name)

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        self.model_dir = model_dir
        self.dim_input = dim_input
        self.hidden_list = hidden_list
        self.resample_every = resample_every
        self.num_masks = num_masks
        self.dir_name = dir_name

        self.device = get_device(device)

        self.made = MADENet(
            dim_input, hidden_list, dim_input, num_masks,
            natural_ordering=False
        )

        self.made.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.made.parameters(), 3e-4, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.1
        )

        if is_load:
            self.load_model(self.model_dir)

    def get_log_prob(self, inputs, n_samples=1):
        """
        args:
            inputs: tensor, shape [batch_size, attr_vocab_size]

        returns:
            log_probs: tensor, shape [batch_size]
                       log-probability of inputs
        """
        with torch.no_grad():
            loss, logits = self.run_batch(
                inputs.numpy(), testing=True, n_samples=n_samples)

        probs = logits.sigmoid().cpu()
        return (probs * inputs + (1-probs) * (1-inputs)).log().sum(-1)

    def train(self, epochs, batch_size, data):
        for idx in range(epochs):
            self.scheduler.step(idx)
            epoch_loss = 0
            batch_amount = 0
            np.random.shuffle(data)
            steps = len(data) // batch_size
            pbar = tqdm(range(steps))
            for _, step in enumerate(pbar):
                if step % self.resample_every == 0:
                    self.made.update_masks()
                batch = data[step*batch_size:(step+1)*batch_size]
                batch_loss, batch_logits = self.run_batch(batch, testing=False)
                epoch_loss += batch_loss.item()
                batch_amount += 1
                pbar.set_postfix(Loss="{:.5f}".format(epoch_loss / batch_amount))

            epoch_loss /= batch_amount
            print_time_info("Epoch {} finished, training loss {}".format(
                    idx, epoch_loss))

            print_time_info("Epoch {}: save model...".format(idx))
            self.save_model(self.model_dir)

    def test(self, batch_size, data, n_samples=1):
        with torch.no_grad():
            test_loss = 0
            batch_amount = 0
            steps = len(data) // batch_size
            pbar = tqdm(range(steps))
            for _, step in enumerate(pbar):
                batch = data[step*batch_size:(step+1)*batch_size]
                batch_loss, batch_logits = self.run_batch(
                        batch, testing=True, n_samples=n_samples)
                test_loss += batch_loss.item()
                batch_amount += 1

            test_loss /= batch_amount
            print_time_info("testing finished, testing loss {}".format(test_loss))

    def run_batch(self, batch, testing=False, n_samples=1):
        if testing:
            self.made.eval()
        else:
            self.made.train()

        inputs = torch.from_numpy(batch).to(self.device)
        xbhat = torch.zeros_like(inputs)

        for _ in range(n_samples):
            if testing:
                self.made.update_masks()
            logits = self.made(inputs)
            xbhat += logits
        xbhat /= n_samples

        batch_size = batch.shape[0]
        loss = F.binary_cross_entropy_with_logits(xbhat, inputs, reduction='sum') / batch_size

        if not testing:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, xbhat

    def save_model(self, model_dir):
        path = os.path.join(model_dir, "made.ckpt")
        torch.save(self.made, path)
        print_time_info("Save model successfully")

    def load_model(self, model_dir):
        path = os.path.join(model_dir, "made.ckpt")
        if not os.path.exists(path):
            print_time_info("Loading failed, start training from scratch...")
        else:
            self.made = torch.load(path, map_location=self.device)
            print_time_info("Load model from {} successfully".format(model_dir))

    @classmethod
    def load_pretrained(cls,
                        model_dir,
                        device=None):
        config_path = os.path.join(model_dir, "made_config")
        args = dict()
        for line in open(config_path):
            name, value = line.strip().split(': ', maxsplit=1)
            args[name] = value

        made = cls(
                dim_input=int(args['dim_input']),
                hidden_list=list(map(int, args["hiddens"].split(','))),
                resample_every=int(args['resample_every']),
                num_masks=int(args['num_masks']),
                model_dir=args['model_dir'],
                is_load=True,
                device=device,
                dir_name=args['dir_name']
        )
        return made
