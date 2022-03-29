git clone https://github.com/google-research/bert

pip install skip-thoughts
PRETRAINED_MODELS_DIR="skip_thoughts/pretrained/"

mkdir -p ${PRETRAINED_MODELS_DIR}
cd ${PRETRAINED_MODELS_DIR}

# Download and extract the unidirectional model.
wget "http://download.tensorflow.org/models/skip_thoughts_uni_2017_02_02.tar.gz"
tar -xvf skip_thoughts_uni_2017_02_02.tar.gz
rm skip_thoughts_uni_2017_02_02.tar.gz

# Download and extract the bidirectional model.
wget "http://download.tensorflow.org/models/skip_thoughts_bi_2017_02_16.tar.gz"
tar -xvf skip_thoughts_bi_2017_02_16.tar.gz
rm skip_thoughts_bi_2017_02_16.tar.gz


if permision error:
export TFHUB_CACHE_DIR=/my_module_cache

tensorflow-gpu=1.12.0
pip install tensorflow-hub
cuda 9
CUDA Driver Version: 396.26 
conda install cudatoolkit=10.0
