import argparse
import pickle
from model_lm import LM
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
from argument import define_arguments

_, args = define_arguments()

args = add_path(args)
'''
if args.verbose_level > 0:
    print_config(args)
'''
print("-----")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_engine = DataEngine(data_dir=args.data_dir, data_split="train", with_intent=args.with_intent)
test_data_engine = DataEngine(data_dir=args.data_dir, data_split="test", with_intent=args.with_intent)

# vocab, rev_vocab, token_vocab, rev_token_vocab = \
#         pickle.load(open(args.vocab_file, 'rb'))
# attr_vocab_size = len(token_vocab)
# vocab_size = args.vocab_size + 4

model = LM(
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        train_data_engine=train_data_engine,
        test_data_engine=test_data_engine,
        dim_hidden=args.hidden_size,
        dim_embedding=args.embedding_dim,
        vocab_size=train_data_engine.tokenizer.get_vocab_size(),
        n_layers=args.n_layers,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        is_load=args.is_load,
        replace_model=args.replace_model,
        device=device,
        dir_name=args.dir_name
)

# record model config
if not args.is_load:
    with open(os.path.join(model.model_dir, "lm_config"), "w+") as f:
        for arg in vars(args):
            f.write("{}: {}\n".format(
                arg, str(getattr(args, arg))))
        f.close()

loss_weight = np.ones(train_data_engine.tokenizer.get_vocab_size())
loss_weight[_PAD] = args.padding_loss
loss_weight[_EOS] = args.eos_loss
loss_weight = torch.tensor(loss_weight, dtype=torch.float)
loss_func = nn.CrossEntropyLoss(weight=loss_weight.cuda(), ignore_index=-1)

if args.train:
    try:
        model.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                criterion=loss_func,
                save_epochs=args.save_epochs)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
else:
    model.test(
            batch_size=args.batch_size,
            criterion=loss_func,
            result_path=os.path.join(model.log_dir, "validation", "test.txt"))
