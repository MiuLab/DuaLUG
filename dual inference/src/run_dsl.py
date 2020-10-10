import argparse
import pickle
from model_dsl import DSL
from model_lm import LM
from model_marginal import Marginal
from module_dsl import DSLCriterion
from data_engine_nlu_nlg import DataEngine
from text_token import _UNK, _PAD, _BOS, _EOS
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import traceback
import pdb
from utils import print_config, add_path
from model_utils import get_embeddings
from argument import define_arguments
from utils import get_time
from module_dim import Criterion

_, args = define_arguments()

args = add_path(args)

print("-----")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_engine = DataEngine(data_dir=args.data_dir, data_split="train", with_intent=args.with_intent)
test_data_engine = DataEngine(data_dir=args.data_dir, data_split="test", with_intent=args.with_intent)

vocab_size = train_data_engine.tokenizer.get_vocab_size()
# just a random number 
attr_vocab_size = 10

model = DSL(
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        train_data_engine=train_data_engine,
        test_data_engine=test_data_engine,
        dim_hidden=args.hidden_size,
        dim_embedding=args.embedding_dim,
        vocab_size=train_data_engine.tokenizer.get_vocab_size(),
        attr_vocab_size=attr_vocab_size,
        n_layers=args.n_layers,
        bidirectional=args.bidirectional,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        is_load=args.is_load,
        replace_model=args.replace_model,
        device=device,
        dir_name=args.dir_name,
        with_intent=args.with_intent
)

lm = None
if args.lm_model_dir:
    lm = LM.load_pretrained(
        args.lm_model_dir,
        train_data_engine,
        test_data_engine,
        device
    )

made = None
if args.made_model_dir:
    # made = Marginal.load_pretrained(
    #     args.made_model_dir,
    #     device=device
    # )
    made = Marginal(
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
    made.maskpredict = torch.load(os.path.join(args.made_model_dir, "maskpredict.ckpt"))
        # (f"../data/model/{args.dataset}_maskpredict_2/maskpredict.ckpt")

# record model config
if not args.is_load:
    with open(os.path.join(model.log_dir, "model_config"), "w+") as f:
        for arg in vars(args):
            f.write("{}: {}\n".format(
                arg, str(getattr(args, arg))))
        f.close()

loss_weight = np.ones(train_data_engine.tokenizer.get_vocab_size())
loss_weight[_PAD] = args.padding_loss
loss_weight[_EOS] = args.eos_loss
loss_weight = torch.tensor(loss_weight, dtype=torch.float)
loss_func = DSLCriterion(
    loss_weight,
    pretrain_epochs=args.pretrain_epochs,
    LM=lm,
    MADE=made,
    lambda_xy=args.lambda_xy,
    lambda_yx=args.lambda_yx,
    made_n_samples=args.made_n_samples,
    propagate_other=args.propagate_other,
    with_intent=(args.dataset != "e2enlg")
)
loss_func_nlg = Criterion(
    "nlg",
    args.nlg_reward_type,
    loss_weight,
    supervised=(args.supervised == 1),
    rl_lambda=args.nlg_reward_lambda,
    rl_alpha=args.rl_alpha,
    pretrain_epochs=args.pretrain_epochs,
    total_epochs=args.epochs,
    anneal_type=args.sup_anneal_type,
    LM=lm
)
slot_loss_weight = np.ones(len(train_data_engine.nlu_slot_vocab))
slot_loss_weight = torch.tensor(slot_loss_weight, dtype=torch.float)
loss_func_nlu = Criterion(
    "nlu",
    args.nlu_reward_type,
    slot_loss_weight,
    supervised=(args.supervised == 1),
    rl_lambda=args.nlu_reward_lambda,
    rl_alpha=args.rl_alpha,
    pretrain_epochs=args.pretrain_epochs,
    total_epochs=args.epochs,
    anneal_type=args.sup_anneal_type,
    training_set_label_samples=train_data_engine.training_set_label_samples,
    MADE=made
)
if args.train:
    try:
        model.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                criterion=loss_func,
                criterion_nlg=loss_func_nlg,
                criterion_nlu=loss_func_nlu,
                save_epochs=args.save_epochs,
                teacher_forcing_ratio=args.teacher_forcing_ratio,
                tf_decay_rate=args.tf_decay_rate,
                max_norm=args.max_norm)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
else:
    model.test(
            batch_size=args.batch_size,
            criterion=loss_func)
