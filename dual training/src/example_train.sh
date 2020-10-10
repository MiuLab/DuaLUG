#!/bin/bash

# Train language model
python3 run_lm.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 \
    --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU \
    --n_layers 2 --hidden_size 300  --embedding_dim 100 --epochs 1 --batch_size 32 --optimizer Adam \
    --learning_rate 0.001 --padding_loss 0.0 --eos_loss 1.0 --verbose_level 1 --save_epochs 1 \
    --is_load 0 --train 1 --dir_name lm

# Train made
python3 create_made_data.py ../data/E2ENLG_{train,valid}_data.pkl ../data/E2ENLG_vocab.pkl ../data/E2ENLG_made.npz
python3 run_made.py -q 500 -n 10 -r 20 -s 10 --model_dir ../data/model_slt/ --dir_name made --train ../data/E2ENLG_made.npz

# Dual supervised learning with lambda=0.1
python3 run_dsl.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --use_embedding 0 --regen 0 --replace_model 0 \
    --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 \
    --hidden_size 200  --embedding_dim 50 --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam \
    --learning_rate 0.001 --teacher_forcing_ratio 0.9 --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 \
    --is_load 0 --train 1 --dir_name dsl-l0.1 --lm_model_dir ../data/model_slt/lm --made_model_dir ../data/model_slt/made \
    --lambda_xy 0.1 --lambda_yx 0.1

# Joint supervised learning (nlu & nlg both use straight-through)
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-nlu_st-nlg_st --nlu_st 1 --nlg_st 1 --primal_supervised 1 --dual_supervised 1 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint supervised learning (nlu uses straight-through, nlg does not)
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-nlu_st-nlg_nost --nlu_st 1 --nlg_st 0 --primal_supervised 1 --dual_supervised 1 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint supervised learning (nlg uses straight-through, nlu does not)
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-nlu_nost-nlg_st --nlu_st 0 --nlg_st 1 --primal_supervised 1 --dual_supervised 1 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint supervised learning (neither nlg nor nlu uses straight-through)
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-nlu_nost-nlg_nost --nlu_st 0 --nlg_st 0 --primal_supervised 1 --dual_supervised 1 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint supervised learning + RL_mid (f1 & bleu+rough)
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-f1-bleurouge-mid --nlu_st 0 --nlg_st 0 --primal_supervised 1 --dual_supervised 1 --pretrain_epochs 3 \
    --nlu_reward_type f1 --nlu_reward_lambda 0.05 --nlg_reward_type bleu-rouge --nlg_reward_lambda 0.05 --mid_sample_size 5 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint supervised learning + RL_end (f1 & bleu+rough)
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-f1-bleurouge-end --nlu_st 0 --nlg_st 0 --primal_supervised 1 --dual_supervised 1 --pretrain_epochs 3 \
    --nlu_reward_type f1 --nlu_reward_lambda 0.05 --nlg_reward_type bleu-rouge --nlg_reward_lambda 0.05 --dual_sample_size 5 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint supervised learning + RL_mid (loss)
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-loss-mid --nlu_st 0 --nlg_st 0 --primal_supervised 1 --dual_supervised 1 --pretrain_epochs 3 \
    --nlu_reward_type loss --nlu_reward_lambda 0.001 --nlg_reward_type loss --nlg_reward_lambda 0.0001 --mid_sample_size 5 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint supervised learning + RL_end (loss)
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-loss-end --nlu_st 0 --nlg_st 0 --primal_supervised 1 --dual_supervised 1 --pretrain_epochs 3 \
    --nlu_reward_type loss --nlu_reward_lambda 0.001 --nlg_reward_type loss --nlg_reward_lambda 0.0001 --dual_sample_size 5 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint supervised learning + RL_mid (made & lm)
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-made-lm-mid --nlu_st 0 --nlg_st 0 --primal_supervised 1 --dual_supervised 1 --pretrain_epochs 3 \
    --nlu_reward_type made --nlu_reward_lambda 0.00005 --nlg_reward_type lm --nlg_reward_lambda 0.00001 --mid_sample_size 5 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint supervised learning + RL_end (made & lm)
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-made-lm-end --nlu_st 0 --nlg_st 0 --primal_supervised 1 --dual_supervised 1 --pretrain_epochs 3 \
    --nlu_reward_type made --nlu_reward_lambda 0.00005 --nlg_reward_type lm --nlg_reward_lambda 0.00001 --dual_sample_size 5 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint semi-supervised learning
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-semi --nlu_st 0 --nlg_st 0 --primal_supervised 1 --dual_supervised 1 \
    --schedule semi --supervised 1 --model nlu-nlg

# Joint unsupervised learning
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-unsup --nlu_st 0 --nlg_st 0 --primal_supervised 0 --dual_supervised 1 \
    --schedule joint --supervised 1 --model nlu-nlg

# Joint unsupervised learning + RL
python3 main.py \
    --data_dir ../data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name joint-unsup-rl --nlu_st 0 --nlg_st 0 --primal_supervised 0 --dual_supervised 1 --primal_reinforce 1 --dual_reinforce 0 \
    --nlu_reward_type made --nlu_reward_lambda 0.00005 --nlg_reward_type lm --nlg_reward_lambda 0.00001 --mid_sample_size 5 \
    --schedule joint --supervised 1 --model nlu-nlg
