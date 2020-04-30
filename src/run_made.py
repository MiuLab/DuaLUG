import argparse
import pickle
from model_made import MADE
import numpy as np
import torch
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help="Path to data npz")
parser.add_argument('-q', '--hiddens', type=str, default='500', help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
parser.add_argument('-n', '--num_masks', type=int, default=1, help="Number of orderings for order/connection-agnostic training")
parser.add_argument('-r', '--resample_every', type=int, default=20, help="For efficiency we can choose to resample orders/masks only once every this many steps")
parser.add_argument('-s', '--samples', type=int, default=1, help="How many samples of connectivity/masks to average logits over during inference")
parser.add_argument('--model_dir', type=str, default='../model')
parser.add_argument('--dir_name', type=str, default='made')
parser.add_argument('--train', action='store_true')
parser.add_argument('--load', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading data from", args.data_path)
data = np.load(args.data_path)
train_data, test_data = data['train_data'], data['valid_data']

hidden_list = list(map(int, args.hiddens.split(',')))

model = MADE(
        dim_input=train_data.shape[1],
        hidden_list=hidden_list,
        resample_every=args.resample_every,
        num_masks=args.num_masks,
        model_dir=args.model_dir,
        is_load=args.load,
        device=device,
        dir_name=args.dir_name
)

# record model config
if not args.load:
    with open(os.path.join(model.model_dir, "made_config"), "w+") as f:
        for arg in vars(args):
            f.write("{}: {}\n".format(
                arg, str(getattr(args, arg))))
        f.write("{}: {}\n".format('dim_input', train_data.shape[1]))
        f.close()

if args.train:
    model.train(
        epochs=10,
        batch_size=100,
        data=train_data
    )
else:
    model.test(
        batch_size=100,
        data=test_data,
        n_samples=args.samples
    )

    model = MADE.load_pretrained(model.model_dir, device)
    model.test(
        batch_size=64,
        data=test_data,
        n_samples=args.samples
    )
