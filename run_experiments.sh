#!/bin/bash
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3.6
for i in gastrofy SST ger movie provenword aclImdb cz
do
	echo "---> $i"
        echo "word"
        python test.py --data datasets/${i}/${i}.train --dev datasets/${i}/${i}.dev  --predict datasets/${i}/${i}.test --emb_size_word 128 --save model > results/${i}.word 2>&1
        rm *.hdf5
        echo "bert"
        python test.py --data datasets/${i}/${i}.train --dev datasets/${i}/${i}.dev  --predict datasets/${i}/${i}.test --bert True --save model > results/${i}.bert  2>&1
        rm *.hdf5
        echo "char"
        python test.py --data datasets/${i}/${i}.train --dev datasets/${i}/${i}.dev  --predict datasets/${i}/${i}.test --emb_size_char 128 --save model > results/${i}.char 2>&1
        rm *.hdf5
        echo "bpe"
        python test.py --data datasets/${i}/${i}.train --dev datasets/${i}/${i}.dev  --predict datasets/${i}/${i}.test --emb_size_bpe 128 --save model > results/${i}.bpe 2>&1
        rm *.hdf5
        python test.py --data datasets/${i}/${i}.train --dev datasets/${i}/${i}.dev  --predict datasets/${i}/${i}.test --glove word --save model > results/${i}.glove_word 2>&1
        echo "glove_word"
        python test.py --data datasets/${i}/${i}.train --dev datasets/${i}/${i}.dev  --predict datasets/${i}/${i}.test --glove bpe --save model > results/${i}.glove_bpe 2>&1
        echo "glove_char"
        python test.py --data datasets/${i}/${i}.train --dev datasets/${i}/${i}.dev  --predict datasets/${i}/${i}.test --glove char --save model > results/${i}.glove_char 2>&1
        echo "pv"
        python test.py --data datasets/${i}/${i}.train --dev datasets/${i}/${i}.dev  --predict datasets/${i}/${i}.test --pv True --save model > results/${i}.glove_word 2>&1
done

