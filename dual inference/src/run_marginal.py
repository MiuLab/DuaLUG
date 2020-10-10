import torch
import torch.nn as nn
import numpy as np
import os
import sys
import traceback
import pdb
import argparse
import pickle
from model_dual_dim import Dual
from model_lm import LM
from model_marginal import Marginal
from data_engine_nlu_nlg import DataEngine
from data_engine_nlu import DataEngine as DataEngineNLU
from data_engine_nlg import DataEngine as DataEngineNLG
from text_token import _UNK, _PAD, _BOS, _EOS
from utils import print_config, add_path
from model_utils import get_embeddings
from argument import define_arguments
from utils import get_time

_, args = define_arguments()

args = add_path(args)
'''
if args.verbose_level > 0:
    print_config(args)
'''
print("-----")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("loading data from", args.data_path)
# data = np.load(args.data_path)
# train_data, test_data = data['train_data'], data['valid_data']

# hidden_list = list(map(int, args.hiddens.split(',')))

train_data_engine = DataEngine(data_dir=args.data_dir, data_split="train", with_intent=args.with_intent)
test_data_engine = DataEngine(data_dir=args.data_dir, data_split="test", with_intent=args.with_intent)
vocab_size = train_data_engine.tokenizer.get_vocab_size()

model = Marginal(
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        train_data_engine=train_data_engine,
        test_data_engine=test_data_engine,
        dim_hidden=args.hidden_size,
        dim_embedding=args.embedding_dim,
        vocab_size=vocab_size,
        n_layers=args.n_layers,
        bidirectional=args.bidirectional,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        is_load=args.is_load,
        replace_model=args.replace_model,
        model=args.model,
        schedule=args.schedule,
        device=device,
        dir_name=args.dir_name,
        f1_per_sample=("f1" in args.nlu_reward_type),
        dim_loss=args.dim_loss,
        with_intent = args.with_intent,
        nlg_path=f"../data/model/{args.dataset}_baseline_2/nlg.ckpt"
)

# record model config
if not args.is_load:
    with open(os.path.join(model.model_dir, "marginal_config"), "w+") as f:
        configs = [
            f"dim_embedding: {args.embedding_dim}",
            f"dim_hidden: {args.hidden_size}",
            f"vocab_size: {vocab_size}",
            f"n_slot_key: {len(train_data_engine.nlg_slot_vocab)}",
            f"n_intent: {len(train_data_engine.intent_vocab)}",
            f"n_layers: {args.n_layers}",
            f"bidirectional: {args.bidirectional}",
            f"batch_size: {args.batch_size}"
        ]
        for line in configs:
            f.write("{}\n".format(line))
        f.close()

if args.train:
    model.maskpredict.cuda()
    model.train(
        epochs=10,
        batch_size=32
    )
else:
    model.maskpredict.cuda()
    model.test(
        batch_size=32,
    )

    model = Marginal.load_pretrained(model.model_dir, device)
    model.test(
        batch_size=64,
        data=test_data,
        n_samples=args.samples
    )
