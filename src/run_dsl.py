import argparse
import pickle
from model_dsl import DSL
from model_lm import LM
from model_made import MADE
from module_dsl import DSLCriterion
from data_engine import DataEngine
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

_, args = define_arguments()

args = add_path(args)

print("-----")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_engine = DataEngine(
    data_dir=args.data_dir,
    dataset=args.dataset,
    save_path=args.train_data_file,
    vocab_path=args.vocab_file,
    is_spacy=args.is_spacy,
    is_lemma=args.is_lemma,
    fold_attr=args.fold_attr,
    use_punct=args.use_punct,
    vocab_size=args.vocab_size,
    n_layers=args.n_layers,
    en_max_length=(args.en_max_length if args.en_max_length != -1 else None),
    de_max_length=(args.de_max_length if args.de_max_length != -1 else None),
    regen=args.regen,
    train=True
)

test_data_engine = DataEngine(
    data_dir=args.data_dir,
    dataset=args.dataset,
    save_path=args.valid_data_file,
    vocab_path=args.vocab_file,
    is_spacy=args.is_spacy,
    is_lemma=args.is_lemma,
    fold_attr=args.fold_attr,
    use_punct=args.use_punct,
    vocab_size=args.vocab_size,
    n_layers=args.n_layers,
    en_max_length=(args.en_max_length if args.en_max_length != -1 else None),
    de_max_length=(args.de_max_length if args.de_max_length != -1 else None),
    regen=args.regen,
    train=False
)

vocab, rev_vocab, token_vocab, rev_token_vocab = \
        pickle.load(open(args.vocab_file, 'rb'))
attr_vocab_size = len(token_vocab)
vocab_size = args.vocab_size + 4


model = DSL(
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        train_data_engine=train_data_engine,
        test_data_engine=test_data_engine,
        dim_hidden=args.hidden_size,
        dim_embedding=args.embedding_dim,
        vocab_size=vocab_size,
        attr_vocab_size=attr_vocab_size,
        n_layers=args.n_layers,
        bidirectional=args.bidirectional,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        is_load=args.is_load,
        replace_model=args.replace_model,
        device=device,
        dir_name=args.dir_name
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
    made = MADE.load_pretrained(
        args.made_model_dir,
        device=device
    )

# record model config
if not args.is_load:
    with open(os.path.join(model.log_dir, "model_config"), "w+") as f:
        for arg in vars(args):
            f.write("{}: {}\n".format(
                arg, str(getattr(args, arg))))
        f.close()

loss_weight = np.ones(args.vocab_size + 4)
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
    propagate_other=args.propagate_other
)

if args.train:
    try:
        model.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                criterion=loss_func,
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
