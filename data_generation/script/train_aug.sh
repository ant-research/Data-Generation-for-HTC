#!/bin/bash

# pip install -r requirements.txt

GPUID=0,1
RANDOM_SEED=123
DATASET=WebOfScience
LM_STEP=10000
MAX_LENGTH=512


SAMPLE_WITH_LABELS=1
EVAL_REPEAT_NUM=1
EVAL_SAMPLE_NUM=5
OVERSAMPLE=100

DATA_ROOT=./NLU_training_dataset/$DATASET
CONFIG_ROOT=./model_config/$DATASET
# OUTPUT_ROOT=$DATASET\_$RANDOM_SEED\_tag_keyword_two_prefix_$LM_STEP\_step
# OUTPUT_ROOT=label_keyword_512
OUTPUT_ROOT=keyword_label_512

mkdir $OUTPUT_ROOT
pre_trained_model=t5-base

PRETRAINED_PT_LM=./pretrain_web_page_keyword_t5_short

cp -r ./nltk_data /root/nltk_data



CUDA_VISIBLE_DEVICES=$GPUID python trian/train_data_agumentation.py --config $CONFIG_ROOT/nlg_prefix.yml \
--config-override checkpoint_every_step 500 num_training_steps $LM_STEP select_model_by_ppl True load_from_pretrained True training_da_mode "['keyword']" eval_da_mode "['keyword']" max_length $MAX_LENGTH train_path NLU_training_dataset/WebOfScience/train_whole.txt random_seed $RANDOM_SEED dev_path NLU_training_dataset/WebOfScience/train_whole.txt enable_full_finetune True \
--train --serialization-dir $OUTPUT_ROOT/nlg_model_mix \
--pre_trained_model $pre_trained_model\
# --start-from-checkpoint $PRETRAINED_PT_LM

# CUDA_VISIBLE_DEVICES=$GPUID python train_data_agumentation.py --config $CONFIG_ROOT/nlg_prefix.yml \
# --config-override eval_da_mode "['keyword']" max_length $MAX_LENGTH train_path NLU_training_dataset/WebOfScience/train_whole.txt random_seed $RANDOM_SEED test_path NLU_training_dataset/WebOfScience/train_whole.txt eval_data_replication $EVAL_REPEAT_NUM sample_num $EVAL_SAMPLE_NUM enable_filtering_error True  enable_full_finetune True \
# --start-from-checkpoint $OUTPUT_ROOT/nlg_model_mix \
# --test \
# --output-path $OUTPUT_ROOT/uniform_train_keyword_label_beam5.txt 

