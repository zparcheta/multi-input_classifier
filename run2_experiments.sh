#!/bin/bash
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh

conda activate p3.6_classifier
for i in gastrofy
do
  for filter in 3 3,4,5
  do
    for red in CNN RNN LSTM GRU
    do
        echo "---> $i $red $filter" 
        #echo "word"
        python test.py --data datasets/${i}/${i}.train --dev datasets/${i}/${i}.dev  --predict datasets/${i}/${i}.test --emb_size_bpe 128 --filter_sizes $filter --network $red --save model > ${i}_${filter}_${red}.txt

    done
  done
done
