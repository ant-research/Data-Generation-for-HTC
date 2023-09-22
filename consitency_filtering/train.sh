nvidia-smi

# pip install transformers
pip install torch-geometric
pip install fairseq==0.10.0
pip install yake
pip install -r requirements.txt

# pip uninstall tensorflow --yes
# mkdir checkpoints



SEED=12
# WebOfScience,nyt,rcv1
# train_all_lambada_512,train_all_promda_all,train_all_gda,label_keyword_uniform_hier_v4_20_beam5,eda_train_whole,bt_train_whole,label_keyword_uniform_hier_v4_20_beam5_1_times,label_keyword_uniform_hier_v4_20_base_beam5
# train_gda_beam5,train_lambada_beam5,label_keyword_uniform_hier_v4_15_nyt_beam5,train_promda_all_beam5,train_eda_whole,bt_train_whole
task=WebOfScience

filter_data=train_all_lambada_2times_512
OUT_DIR=${task}_consistency_filtering_es6_bert_${filter_data}_${SEED}
# OUT_DIR=${task}_es6_bert_${filter_data}_${SEED}

mkdir $OUT_DIR

mkdir ${OUT_DIR}_data



python download.py \
      --oss_model_path=checkpoint_best_keyword.pt \

python download.py \
      --oss_model_path=MULTI_LABEL_DA/nltk_data.tar.gz \


cp -r ./nltk_data /root/nltk_data


# python download.py \
#       --oss_model_path=HTC_contrasitive/checkpoints.tar.gz \



# python download.py \
#       --oss_model_path=HTC/fairseq-main.zip \


# unzip fairseq-main.zip

# cd ./fairseq-main


# pip install --editable ./


# cd ..

python download.py \
      --oss_model_path=HTC/torch_scatter-2.0.8-cp36-cp36m-linux_x86_64.whl \

python download.py \
      --oss_model_path=HTC/torch_sparse-0.6.12-cp36-cp36m-linux_x86_64.whl \


pip install torch_scatter-2.0.8-cp36-cp36m-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp36-cp36m-linux_x86_64.whl

# python download.py \
#       --oss_model_path=HTC/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl \

# python download.py \
#       --oss_model_path=HTC/torch_sparse-0.6.14-cp38-cp38-linux_x86_64.whl \


# pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
# pip install torch_sparse-0.6.14-cp38-cp38-linux_x86_64.whl
# pip install torch-scatter
# pip install torch-sparse

python download.py \
      --oss_model_path=HTC/data.tar.gz \

python download.py \
      --oss_model_path=bert-base-uncased.tar.gz \

python download.py \
      --oss_model_path=MULTI_LABEL_DA/NLU_training_dataset.tar.gz \




if [[ $task == 'WebOfScience' ]]; then
    cd data/WebOfScience

    python data_wos.py

    cd ..
    cd ..
fi


python download.py \
      --oss_model_path=HTC/${task}/checkpoints.tar.gz \
      

# python train.py --name test --batch 12 --data WebOfScience --lamb 0.1 --thre 0.02 --seed $SEED




python filter.py --name ${task}-test --input_dir NLU_training_dataset/${task}/${filter_data}.txt --output_dir ${OUT_DIR}_data/filter_0.txt



i=1

# for i in 1 2 3
# do
	# NLU model run for consistency filtering

rm -r checkpoints

mkdir checkpoints

python concentrate_data.py --input_dir NLU_training_dataset/${task}/train_whole.txt ${OUT_DIR}_data/filter_$[$i-1].txt --output_dir ${OUT_DIR}_data/concat_${i}.txt

# python concentrate_data.py --input_dir NLU_training_dataset/${task}/train_whole.txt NLU_training_dataset/${task}/${filter_data}.txt --output_dir ${OUT_DIR}_data/concat_${i}.txt

# python train_aug.py --name test --batch 12 --data ${task} --lamb 0.1 --thre 0.02 --seed $SEED --input_dir ${OUT_DIR}_data/concat_${i}.txt

python train_aug.py --name test --batch 12 --data ${task} --lamb 0.3 --thre 0.005 --seed $SEED --input_dir ${OUT_DIR}_data/concat_${i}.txt

# python train_aug.py --name test --batch 12 --data ${task} --lamb 0.1 --thre 0.02 --seed $SEED --input_dir NLU_training_dataset/${task}/${filter_data}.txt


mkdir ${OUT_DIR}_data/iter_${i}

python test.py --name ${task}-test --output_dir ${OUT_DIR}/

# python filter.py --name ${task}-test --input_dir ${OUT_DIR}_data/filter_$[$i-1].txt --output_dir ${OUT_DIR}_data/filter_${i}.txt    


# done


python upload.py \
    --output_dir=./${OUT_DIR} \


python upload.py \
    --output_dir=./${OUT_DIR}_data \


# python upload.py \
#       --output_dir=./$OUT_DIR \