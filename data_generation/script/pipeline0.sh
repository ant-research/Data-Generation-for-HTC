nvidia-smi

# pip install transformers
pip install torch-geometric
pip install fairseq==0.10.0

# mkdir checkpoints

SEED=42


filter_data=train_all_label_keyword
OUT_DIR=consistency_filtering_contrasitive_${filter_data}_${SEED}

mkdir $OUT_DIR


pip install torch-scatter
pip install torch-sparse



cd data/WebOfScience

python data_wos.py

cd ..
cd ..

python train.py --name test --batch 12 --data WebOfScience --lamb 0.1 --thre 0.02 --seed $SEED

python filter.py --name WebOfScience-test --input_dir NLU_training_dataset/WebOfScience/${filter_data}.txt --output_dir $OUT_DIR/filter_0.txt


for i in 1 2 3
do
# NLU model run for consistency filtering

rm -r checkpoints

mkdir checkpoints

python concentrate_data.py --input_dir NLU_training_dataset/WebOfScience/train_whole.txt $OUT_DIR/filter_$[$i-1].txt --output_dir $OUT_DIR/concat_${i}.txt

python train_aug.py --name test --batch 12 --data WebOfScience --lamb 0.1 --thre 0.02 --seed $SEED --input_dir $OUT_DIR/concat_${i}.txt

mkdir ${OUT_DIR}/iter_${i}

python test.py --name WebOfScience-test --output_dir ${OUT_DIR}/iter_${i}

python filter.py --name WebOfScience-test --input_dir $OUT_DIR/filter_$[$i-1].txt --output_dir $OUT_DIR/filter_${i}.txt    

done