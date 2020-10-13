# Dual Inference of NLU and NLG
> The implementation of the paper [*"Dual Inference for Improving Language Understanding and Generation"*](https://arxiv.org/abs/2010.04246) (Findings of EMNLP 2020)

## Requirements
* Python >= 3.6
* torch >= 0.4.1
* Other required packages are listed in `requirements.txt`

## Setup
```bash
# use virtualenv or anaconda to create a virtual environment

pip install -r requirements.txt
```

## Usage

- `$DATA` could be either `atis-2/snips/e2enlg`.
- `$LM_MODEL_DIR` is the prepared LM model dir.
- `$MASK_PREDICT_MODEL_DIR` is the prepared model dir for _**masked prediction of semantic labels**_ described in Section 2.2.
- `MODEL_DIR` is the output model (NLG/NLU) dir. 
- Run the **iterative baseline** by assigning `--lambda_xy 0.0 --lambda_yx 0.0` to `run_dsl.py`
- Run the **dual supervised learning** by assigning `--lambda_xy 0.1 --lambda_yx 0.1` to `run_dsl.py`
- Run the **joint learning** using `run_dual_jdl.py`

### Prepare Marginal Estimation Model

#### train LM
```bash
python3 run_lm.py \
    --data_dir ../data/ --dataset $DATA --fold_attr 1 --regen 0 --replace_model 0 \
    --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU \
    --n_layers 2 --hidden_size 300  --embedding_dim 100 --epochs 5 --batch_size 32 --optimizer Adam \
    --learning_rate 0.001 --padding_loss 0.0 --eos_loss 1.0 --verbose_level 1 --save_epochs 1 \
    --is_load 0 --train 1 --dir_name $LM_MODEL_DIR
```

#### train masked prediction of semantic labels
```bash
python run_marginal.py \
    --data_dir ../data/ --dataset $DATA --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 32 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name $MASK_PREDICT_MODEL_DIR --nlu_st 1 --nlg_st 1 --primal_supervised 1 --dual_supervised 1 \
    --schedule iterative --supervised 1 --model nlu-nlg
```

### Training
#### Iterative baseline
> run `run_dsl.py` with lambda=0.0
```bash
python3 run_dsl.py \
    --data_dir ../data/ --dataset $DATA --fold_attr 1 --use_embedding 0 --regen 0 --replace_model 0 \
    --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 \
    --hidden_size 200  --embedding_dim 50 --bidirectional 1  --epochs 10 --batch_size 32 --optimizer Adam \
    --learning_rate 0.001 --teacher_forcing_ratio 0.9 --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 \
    --is_load 0 --train 1 --dir_name $MODEL_DIR --lm_model_dir $LM_MODEL_DIR --made_model_dir $MASK_PREDICT_MODEL_DIR \
    --lambda_xy 0.0 --lambda_yx 0.0 
```

#### Dual Supervised Learning
```
python3 run_dsl.py \
    --data_dir ../data/ --dataset $DATA --fold_attr 1 --use_embedding 0 --regen 0 --replace_model 0 \
    --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 \
    --hidden_size 200  --embedding_dim 50 --bidirectional 1  --epochs 10 --batch_size 32 --optimizer Adam \
    --learning_rate 0.001 --teacher_forcing_ratio 0.9 --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 \
    --is_load 0 --train 1 --dir_name $MODEL_DIR --lm_model_dir $LM_MODEL_DIR --made_model_dir $MASK_PREDICT_MODEL_DIR \
    --lambda_xy 0.1 --lambda_yx 0.01
```

#### Joint Learning
```
python run_jdl.py \
    --data_dir ../data/ --dataset atis-2  --fold_attr 1 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 48 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name $MODEL_DIR --nlu_st 1 --nlg_st 1 --primal_supervised 1 --dual_supervised 1 \
    --schedule joint --supervised 1 --model nlu-nlg

```

### Dual Inference

- Set `--load 1 --train 0` for each command for testing.
- `$DATA` could be either `atis-2/snips/e2enlg`.
- `$LM_MODEL_DIR` is the prepared LM model dir.
- `$MASK_PREDICT_MODEL_DIR` is the prepared model dir for _**masked prediction of semantic labels**_ described in Section 2.2.
- Before running inference, you there should be a trained model by either **iterative baseline / dual supervised learning / joint learning**. 
- `$MODEL_DIR` is the trained model dir for inference.
- Run dual inference by setting `$DUAL_W` and `$MARG_W` between 0-1. Set `$DUAL_W` to 1 to disable the dual cycle model. Set `$MARG_W` to 0 to disable the marginal model (LM & MaskPredict).

```bash
python3 run_dual_inf.py \
    --data_dir ../data/ --dataset $DATA --fold_attr 1 --use_embedding 0 --regen 0 --replace_model 0 \
    --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 \
    --hidden_size 200  --embedding_dim 50 --bidirectional 1  --epochs 10 --batch_size 32 --optimizer Adam \
    --learning_rate 0.001 --teacher_forcing_ratio 0.9 --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 \
    --is_load 1 --train 0 --dir_name $MODEL --lm_model_dir $LM_MODEL_DIR --made_model_dir $MASK_PREDICT_MODEL_DIR --dual_inference_weight $DUAL_W --lm_weight $MARG_W --made_weight $MARG_W
```



## Citation

```
@inproceedings{su2020dual,
  title={Dual Inference for Improving Language Understanding and Generation},
  author={Su, Shang-Yu and Chuang, Yung-Sung and Chen, Yun-Nung},
   booktitle = {Findings of EMNLP 2020},
    year={2020}
}
```


## Resources
* [E2E NLG Dataset](http://www.macs.hw.ac.uk/InteractionLab/E2E/)
* [ATIS-2 & SNIPS Dataset](https://github.com/sz128/slot_filling_and_intent_detection_of_SLU)

