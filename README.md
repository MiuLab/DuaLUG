# Dual Learning of NLU and NLG
> The implementation of the papers [*"Dual supervised learning for natural language understanding and generation"*](https://arxiv.org/abs/1905.06196) (ACL 2019) and [*"Towards Unsupervised Language Understanding and Generation by Joint Dual Learning"*](https://arxiv.org/abs/2004.14710) (ACL 2020)


## Requirements
* Python >= 3.6
* torch >= 0.4.1
* Other required packages are listed in `requirements.txt`

## Setup
```
# Get the E2ENLG dataset (from the link below), and put it under data/E2ENLG/
$ mkdir -p data/E2ENLG/

# use virtualenv or anaconda to create a virtual environment
# install required packages in requirements.txt

# download SpaCy model
$ python3 -m spacy download en
```

## Usage

```
# Iterative baseline
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG \
    --fold_attr 1 --vocab_size 500 \
    --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 \
    --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 \
    --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9  \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 \
    --verbose_level 1 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name iterative \
    --primal_supervised 1 --dual_supervised 1 \
    --schedule iterative --supervised 1 --model nlu-nlg

# Dual Supervised Learning
python3 run_dsl.py \
    --data_dir ../data/ --dataset E2ENLG \
    --fold_attr 1 --vocab_size 500 \
    --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 \
    --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 \
    --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9  \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 \
    --verbose_level 1 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name dsl \
    --model nlu-nlg

# Joint Dual Learning
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG \
    --fold_attr 1 --vocab_size 500 \
    --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 \
    --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 \
    --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9  \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 \
    --verbose_level 1 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-nlu_nost-nlg_nost --nlu_st 0 --nlg_st 0 \
    --primal_supervised 1 --dual_supervised 1 \
    --schedule joint --supervised 1 --model nlu-nlg
```

<b>Please refer to `argument.py` and `example_train.sh` for running more experiments. Set `--load 1 --train 0` for each command for testing.</b>

## References
Main papers to be cited:

```
@inproceedings{su2019dual,
  title={Dual supervised learning for natural language understanding and generation},
  author={Su, Shang-Yu and Huang, Chao-Wei and Chen, Yun-Nung},
 booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}

@inproceedings{su2020joint,
  title={Towards Unsupervised Language Understanding and Generation by Joint Dual Learning},
  author={Su, Shang-Yu and Huang, Chao-Wei and Chen, Yun-Nung},
 booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    year={2020}
}

```


## Resources
* [E2E NLG Dataset](http://www.macs.hw.ac.uk/InteractionLab/E2E/)

