#!/bin/bash

# pip install -r requirements.txt

GPUID=3
RANDOM_SEED=42
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

# OUTPUT_ROOT=label_keyword_class_balance
# OUTPUT_ROOT=label_keyword_uniform
# OUTPUT_ROOT=label_keyword_uniform_hybrid
# OUTPUT_ROOT=label_keyword_uniform_yake15
OUTPUT_ROOT=label_keyword_noise
# OUTPUT_ROOT=label_keyword_uniform_hier_v4_20
# OUTPUT_ROOT=label_keyword_uniform_hier_v4_25_${DATASET}

mkdir $OUTPUT_ROOT

cp -r ./nltk_data /root/nltk_data

PRETRAINED_PT_LM=./pretrain_web_page_keyword_t5_short


# python process_data/get_label_keyword.py


# CUDA_VISIBLE_DEVICES=$GPUID python train/train_generator.py --config $CONFIG_ROOT/nlg_prefix.yml \
# --config-override sampling_type "uniform" gen_templete "" checkpoint_every_step 1000 num_training_steps $LM_STEP select_model_by_ppl True load_from_pretrained True training_da_mode "['keyword']" eval_da_mode "['keyword']" max_length $MAX_LENGTH train_path NLU_training_dataset/${DATASET}/train_whole_noise.txt random_seed $RANDOM_SEED dev_path NLU_training_dataset/${DATASET}/train_whole_noise.txt enable_full_finetune True \
# --train --serialization-dir $OUTPUT_ROOT/nlg_model_mix \
# --pre_trained_model t5-large \
# --wandb


CUDA_VISIBLE_DEVICES=$GPUID python train/train_generator.py --config $CONFIG_ROOT/nlg_prefix.yml \
--config-override test True gen_templete "" eval_da_mode "['keyword']" max_length $MAX_LENGTH train_path train_whole_noise_4.txt random_seed $RANDOM_SEED test_path train_whole_noise_4.txt eval_data_replication $EVAL_REPEAT_NUM sample_num $EVAL_SAMPLE_NUM enable_filtering_error True  enable_full_finetune True \
--start-from-checkpoint $OUTPUT_ROOT/nlg_model_mix \
--test \
--output-path ${OUTPUT_ROOT}/${OUTPUT_ROOT}_noise_beam${EVAL_SAMPLE_NUM}_4.txt \
--pre_trained_model t5-large

# CUDA_VISIBLE_DEVICES=$GPUID python train/train_generator.py --config $CONFIG_ROOT/nlg_prefix.yml \
# --config-override test True gen_templete "" eval_da_mode "['keyword']" max_length $MAX_LENGTH train_path NLU_training_dataset/${DATASET}/train_whole.txt random_seed $RANDOM_SEED test_path NLU_training_dataset/${DATASET}/train_whole.txt eval_data_replication $EVAL_REPEAT_NUM sample_num $EVAL_SAMPLE_NUM enable_filtering_error True  enable_full_finetune True \
# --start-from-checkpoint $OUTPUT_ROOT/nlg_model_mix \
# --test \
# --output-path yake_study.txt \
# --pre_trained_model t5-large