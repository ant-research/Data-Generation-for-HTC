

SEED=42

name=WebOfScience-first-test 

filter_data=keyword_label_512/balance_train_keyword_label_beam5
OUT_DIR=consistency_filtering_contrasitive_${filter_data}_${SEED}

mkdir $OUT_DIR

python train/train_classifier.py --name test --batch 12 --data WebOfScience --lamb 0.1 --thre 0.02 --seed $SEED

python process_data/filter.py --name $name --input_dir ${filter_data}.txt --output_dir $OUT_DIR/filter_0.txt


for i in 1 2 3
do
# NLU model run for consistency filtering

rm -r checkpoints

mkdir checkpoints

python process_data/concentrate_data.py --input_dir NLU_training_dataset/WebOfScience/train_whole.txt $OUT_DIR/filter_$[$i-1].txt --output_dir $OUT_DIR/concat_${i}.txt

python train/train_aug.py --name $name --batch 12 --data WebOfScience --lamb 0.1 --thre 0.02 --seed $SEED --input_dir $OUT_DIR/concat_${i}.txt

mkdir ${OUT_DIR}/iter_${i}

python train/test.py --name $name --output_dir ${OUT_DIR}/iter_${i}

python process_data/filter.py --name $name --input_dir $OUT_DIR/filter_$[$i-1].txt --output_dir $OUT_DIR/filter_${i}.txt    

done