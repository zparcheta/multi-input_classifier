# -*- coding: utf-8 -*-::
"""
Created on Thu Jul 5 15:31:15 2018

Author: Zuzanna Parcheta <zparcheta@sciling.com>
        Germán Sanchis-Trilles <gsanchis@sciling.com>

(c) 2018 Sciling, SL
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import logging
default_logger = logging.getLogger()
default_logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(name)s:%(levelname)s:%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf
import keras
#tf.logging.set_verbosity(tf.logging.ERROR)
import pandas as pd
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Dense, Dropout, Input, Conv2D, MaxPool2D,
                                  Reshape, Flatten, Concatenate, Dot,  LSTM, SimpleRNN, GRU)
import random
from bpetokenizer import BPETokenizer
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Model
import logging
import collections
import gensim.utils
import pickle
import multiprocessing
import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
from gensim.models.doc2vec import Doc2Vec
from glove import Corpus, Glove
from sklearn.metrics import f1_score, average_precision_score, accuracy_score, precision_score
import ml_metrics 
import os
from numpy.random import seed
seed(1)

# cudnn error fixing
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = InteractiveSession(config=config)
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#from tensorflow.compat.v1.keras.layers.embeddings import Embedding
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Initialize session
print('Starting ...')
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

class GloveEmbeding:
  
  def __init__(self, no_components=128):
    self.no_components = no_components
    self.model = Glove(no_components=self.no_components, learning_rate=0.05) 
    
  def train(self, lines): 
    # lines list of lists
    corpus = Corpus() 
    corpus.fit(lines, window=10)
    self.model.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    self.model.add_dictionary(corpus.dictionary)

  def save(self, path):
    self.model.save(path)
     
  def load(self, path):
    self.load(path)
    
  def embeddings(self, sentences, max_len): 
    # paragraph list of tokens
    matrix = np.zeros((len(sentences), max_len, self.no_components))
    for nr, sentence in enumerate(sentences):
      for nr_tok, tok in enumerate(sentence):
        try:
          emb_nr = self.model.dictionary[tok]
          
          matrix[nr,nr_tok,:] = self.model.word_vectors[emb_nr]
        except KeyError:
          print('word \"' + tok.encode(encoding='UTF-8') + '\" not in dictionary')
        except IndexError:
          pass
    return matrix

class ParagraphVector:
  
  def __init__(self, dm=0, vector_size=128):
    self.common_kwargs = dict(sample=0, 
                     workers=multiprocessing.cpu_count(), 
                     negative=5, 
                     hs=0,
                     window=10,
                     alpha=0.05)
    
    self.SentimentDocument = collections.namedtuple('SentimentDocument', 'words tags sentiment')
    self.model = Doc2Vec(dm=0, vector_size=vector_size, **self.common_kwargs)
    
  def train(self, documents,  epochs=30):
    self.model.train(documents, total_examples=len(documents), epochs=epochs)
    
  def build_vocab(self, documents):
    self.model.build_vocab(documents)
    
  def embeddings(self, sentence):
    # sentence: list of tokens
    return self.model.infer_vector(sentence)
  
  def to_document(self, index, text, clase):
    tokens = gensim.utils.to_unicode(text).split()
    return self.SentimentDocument(tokens, [index], clase)

  def extract_documents(self, df):
      for index, row in df.iterrows():
        yield self.to_document(index, row['sentence'],  row['class'])

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling=None,
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs):

        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean", None]:
            raise NameError(
                "Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path, trainable=self.trainable, name=f"{self.name}_module")
        
        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name and not "/pooler/" in var.name]
            trainable_layers = []
        else:
            trainable_layers = []
            """raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )"""

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)


    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
        masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = mul_mask(result, input_mask)
            #raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")
        
        return pooled


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())
    tf.keras.backend.set_session(sess)

class SeqClass():
    """ This class builds a sequence classification model based on Keras.

    Embeddings at the level of characters, bpe, and words are allowed.

    Full example use:
    >>> m = SeqClass(embed_dims={'char': 10})
    >>> m.set_train("examples/toy.csv")
    >>> m.build_model()
    >>> m.train_model()
    >>> m.predict("examples/toy.csv", nbest=3)
    """
    def __init__(self,
                 network_type='default',
                 embed_dims={'char': 0, 'bpe': 0, 'word': 0},
                 pretrained_embeds={'char': None, 'bpe': None, 'word': None},
                 white_space_char='A', delimiter='\t',logger=False, epochs=1000):
        """ Initialize the class """
        self.logger = logger if logger else default_logger
        self.network_type = network_type
        self.embed_dims = embed_dims
        self.pretrained_embeds = pretrained_embeds
        self.max_len = {'char': 0, 'bpe': 0, 'word': 0, 'bert-word':0}
        self.white_space_char = white_space_char
        self.token_types = [x for x in embed_dims if self.embed_dims[x] != 0]
        self.bpe_tokenizer = None
        self.logger.info("Model defined with {} tokens".format(self.token_types))
        self.num_filters = 512
        #self.filter_size = [3,4,5]
        self.bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        self.epochs = epochs
        self.tokenizer_bert = None
        self.model = None
        self.batch_size = 50
        self.trainig_verbose = 0
        #self.network_type = 'RNN'
        self.prediction = False
        self.delimiter = delimiter
        self.limit_length = True
        self.length_limited = 150
        self.sequences={}
        self.pv_model = None
        self.pv_dim = 128
        self.sequences_dev={}
        self.sequences_test={}
        

    def tokenize(self, data, tok_type):
        """ Helper method for tokenizing a string according to a specific token type.

        Returns a list of lists of tokens.

        :param data: an array of strings
        :param tok_type: the type of token. Allowed values are 'char', 'bpe', 'word'.
        """
        if tok_type == 'char':
            tokens = [list(s.lower().replace(' ', self.white_space_char))
                      for s in data]
        elif tok_type == 'bpe':
            if self.bpe_tokenizer is None:
                
                self.logger.info("BPE tokenizer not provided. Building one")
                self.bpe_tokenizer = BPETokenizer()
                self.bpe_tokenizer.set_data(data)
                self.bpe_tokenizer.train_model()
                self.logger.info("BPE tokenizer built")
            tokens = [self.bpe_tokenizer.tokenize(s.lower()) for s in data]
        elif tok_type == 'word':
            tokens = [s.encode(encoding='UTF-8').split() for s in data]
        else:
            raise AttributeError("attempted to tokenize with unkown token type {}"
                                 .format(tok_type))
        return tokens

    def build_vocab(self, data):
        """ Helper method for building the vocabulary of a list of list of tokens

        Returns a dictionary in the form {key: value}, with key the token

        :param data: the list of list of tokens
        """
        return {k: v for v, k in enumerate(set([w for s in data for w in s]))}

    def tokenize_data(self, data):
        """ Helper method for tokenizing data according to the token types defined
        when creating the class.

        Returns a dictionary of list of lists {tok_type: tokenized_data}

        :param data: a list of strings to be tokenized
        """
        tokenized_data = {}
        for t in self.token_types:
            tokens = self.tokenize(data, t)
            tokenized_data[t] = tokens
        return tokenized_data

    def map_token(self, tok, t):
        """ Helper method to map a specific token to its dictionary value

        Returns the value, if found, or 0 otherwise.

        :param tok: the token to be mapped
        :param t: the type of the token ('char', 'bpe', 'word')
        """
        if tok in self.vocabulary[t]:
            return self.vocabulary[t][tok]
        else:
            return 0

    def get_data(self, sequences, labels):
      """
      """
      sequences_mapped = {t: [[self.map_token(tok, t) for tok in s] for s in sequences[t]]
                   for t in self.token_types}
      return  {t: sequence.pad_sequences(sequences_mapped[t], self.max_len[t])
                for t in self.token_types}, to_categorical(labels, num_classes=self.num_labels)

    def get_bert_data(self, sequences, labels):
      dev_text = [" ".join(t.split()[0:self.max_len['bert-word']]) for t in sequences]
      dev_text = np.array(dev_text, dtype=object)[:, np.newaxis]
      # Convert data to InputExample format
      dev_examples = convert_text_to_examples(dev_text, labels)                
      return convert_examples_to_features(self.tokenizer_bert, dev_examples, max_seq_length=self.max_len['bert-word'])
 
    def limit_length_func(lista,lenght=100):     
        return lista[:lenght]
  
    def preprocess_data(self, data_file, mode="train", shuffle=False):
        """ Processes data read from a file and produces a matrix of integers.

        The lines of the file are tokenized according to the token_types defined when
        instancing the class, and then converted to integers according to the dictionary
        built when this method is invoked in "train" mode.

        Returns a matrix of integers, padded according to the longest sequence seen
        in training.

        :param data_file: the csv data file to read from
        :param separator: the separator used for the csv
        :param mode: either "train" or "test". If in train mode, will build the dictionaries
        :param sequence_field: what column the sequences are in the csv file. Defaults to 1
        :param label_field: what column the labels are in the csv file. Defaults to 2
        """

        import math
        train_df = pd.read_csv(data_file,names=['sentence','class'],sep=self.delimiter)
        labels = list(map(int,train_df["class"].values))
        sequences = train_df["sentence"].tolist()
        sequences_tok = self.tokenize_data(sequences)
        if mode == "train": # TODO: This might be better outside this method
            self.vocabulary = {t: self.build_vocab(sequences_tok[t]) for t in self.token_types}
            self.voc_size = {t: len(self.vocabulary[t]) for t in self.token_types}
            # Compute max_lengths
            for t in self.token_types:
                if self.limit_length and max([len(s) for s in sequences_tok[t]]) > self.length_limited:
                    self.max_len[t] = self.length_limited
                else:    
                    self.max_len[t] = max([len(s) for s in sequences_tok[t]])
            # Map labels to one-hot vectors
            self.train_labels = to_categorical(labels)
            # Compute the number of labels
            self.num_labels = len(self.train_labels[0])
            # and compute vocabulary sizes
            self.sequences, _ = self.get_data(sequences_tok, labels)
            if self.bert == True:
                if self.limit_length and max([len(s.split(' ')) for s in sequences]) > self.length_limited:
                    self.max_len['bert-word'] = self.length_limited
                else:
                    self.max_len['bert-word'] = max([len(s.split(' ')) for s in sequences])
                self.tokenizer_bert = create_tokenizer_from_hub_module(self.bert_path)
                (self.sequences['input_ids'], 
                self.sequences['input_masks'], 
                self.sequences['segment_ids'],_) = self.get_bert_data(sequences, labels)
            if self.pv == True:
                self.pv_model = ParagraphVector(vector_size=self.pv_dim)
                alldocs = list(self.pv_model.extract_documents(train_df))            
                self.pv_model.build_vocab(alldocs)
                self.pv_model.train(alldocs, epochs=30)
                self.sequences['ParagraphVector'] = np.zeros((len(alldocs), self.pv_dim))
                for nr,sent in enumerate(alldocs):
                   self.sequences['ParagraphVector'][nr]=self.pv_model.embeddings(sent.words)
            if self.glove != None:
                tokens = self.tokenize(sequences, self.glove)
                if self.limit_length and max([len(s) for s in tokens]) > self.length_limited:
                    self.max_len['glove_'+self.glove] = self.length_limited
                else:
                    self.max_len['glove_'+self.glove] = max([len(s) for s in tokens])
                self.glove_model = GloveEmbeding(no_components=128)
                self.glove_model.train(tokens)
                self.sequences['glove_'+self.glove] = self.glove_model.embeddings(tokens,self.max_len['glove_'+self.glove])
                
            if self.skip_thoughts != False:
                print(data_file+'_skips.h5py')
                self.sequences['skip_thoughts'] = np.array(HDF5Matrix(data_file+'_skip.h5py', 'skips'))
 
        elif mode == "dev": 
            self.sequences_dev, self.dev_labels = self.get_data(sequences_tok, labels)
            if self.bert == True:
              (self.sequences_dev['input_ids'], 
              self.sequences_dev['input_masks'], 
              self.sequences_dev['segment_ids'],_) = self.get_bert_data(sequences, labels)
            if self.pv == True:
                alldocs_dev = list(self.pv_model.extract_documents(train_df)) 
                self.sequences_dev['ParagraphVector'] = np.zeros((len(alldocs_dev), self.pv_dim))
                for nr, sent in enumerate(alldocs_dev):
                   self.sequences_dev['ParagraphVector'][nr]=self.pv_model.embeddings(sent.words)    
            if self.glove != None:
                tokens = self.tokenize(sequences, self.glove)
                self.sequences_dev['glove_'+self.glove] = self.glove_model.embeddings(tokens, self.max_len['glove_'+self.glove])
            if self.skip_thoughts != False:
                self.sequences_dev['skip_thoughts'] = np.array(HDF5Matrix(data_file+'_skip.h5py', 'skips'))
        elif mode == "test":
            self.sequences_test, self.test_labels = self.get_data(sequences_tok, labels)
            
            if self.bert == True:
              (self.sequences_test['input_ids'], 
              self.sequences_test['input_masks'], 
              self.sequences_test['segment_ids'],_) = self.get_bert_data(sequences, labels)
            if self.pv == True:
                alldocs = list(self.pv_model.extract_documents(train_df)) 
                self.sequences_test['ParagraphVector'] = np.zeros((len(alldocs), self.pv_dim)) 
                for nr, sent in enumerate(alldocs):
                   self.sequences_test['ParagraphVector'][nr]=self.pv_model.embeddings(sent.words)  
            if self.glove != None:
                tokens = self.tokenize(sequences, self.glove)
                self.sequences_test['glove_'+self.glove] = self.glove_model.embeddings(tokens,self.max_len['glove_'+self.glove])
            if self.skip_thoughts != False:
                self.sequences_test['skip_thoughts'] = np.array(HDF5Matrix(data_file+'_skip.h5py', 'skips'))

    def set_train(self, data):
        """ This method establishes a training set for further usage in training.

        :param training_data: the csv file to read the data from
        :param separator: the separator used in the csv file
        :param sequence_field: what column the sequences are in the csv file. Defaults to 1
        :param label_field: what column the labels are in the csv file. Defaults to 2
        """
        # Shuffling the training data proves to provide good results for the dev set
        self.preprocess_data(data, mode="train", shuffle=True)
        
    def set_dev(self, data):
        """ This method establishes a training set for further usage in training.

        :param training_data: the csv file to read the data from
        :param separator: the separator used in the csv file
        :param sequence_field: what column the sequences are in the csv file. Defaults to 1
        :param label_field: what column the labels are in the csv file. Defaults to 2
        """
        # Shuffling the training data proves to provide good results for the dev set
        print("setting dev")
        self.preprocess_data(data, mode="dev", shuffle=True)

    def set_test(self, data):
        """ This method establishes a training set for further usage in training.

        :param training_data: the csv file to read the data from
        :param separator: the separator used in the csv file
        :param sequence_field: what column the sequences are in the csv file. Defaults to 1
        :param label_field: what column the labels are in the csv file. Defaults to 2
        """
        # Shuffling the training data proves to provide good results for the dev set
        print("setting test")
        self.preprocess_data(data, mode="test", shuffle=True)


    def build_model(self, num_filters=512, dropout=0.5):
        """ This method builds the keras model itself.

        Currently, only one model type is available.
        FUNCTIONAL MODEL
        :param filter_sizes: filters for the convolutional layers. Defaults to [3, 4, 5]
        :param num_filters: number of filters for the convolutional layers. Defaults to 512
        :param dropout: the dropout rate. Defaults to 0.5
        """
        inputs = []
        input_layers = {}
        embed_layers = {}
        reshape_layers = {}
        conv_layers = {}
        maxp_layers = {}
        concatenated_tensor = None
        for t in self.token_types:
            if self.max_len[t] == 0 or self.voc_size[t] == 0:
                raise AttributeError("Attempt to build model without setting training data.")
            embeds = None
            trainable = True
            # Read the embeddings, if they are provided
            if t in self.pretrained_embeds and self.pretrained_embeds[t] is not None:
                embeds = self.read_embeddings(self.pretrained_embeds[t])
                embeds = [self.filter_embeddings(embeds, self.vocabulary[t])]
                trainable = False
            # Build the input layer
            input_layers[t] = Input(shape=(self.max_len[t],),
                                    dtype='int32', name=t)
            # Build the embedding layer
            embed_layers[t] = tf.compat.v1.keras.layers.Embedding(input_dim=self.voc_size[t] + 1,
                                       output_dim=self.embed_dims[t],
                                       input_length=self.max_len[t],
                                       weights=embeds,
                                       trainable=trainable)(input_layers[t])
            # TODO: Why is this needed?
            reshape_layers[t] = Reshape((self.max_len[t],
                                         self.embed_dims[t], 1))(embed_layers[t])
                        # Now build three convolutional layers
            conv_layers[t] = []
            maxp_layers[t] = []
            for i in range(len(self.filter_size)):
              if self.network_type == 'CNN':
                # First the convolutional layer itself
                conv_layers[t].append(
                    Conv2D(num_filters,
                           kernel_size=(self.filter_size[i], self.embed_dims[t]),
                           padding='valid', kernel_initializer='normal',
                           activation='relu')(reshape_layers[t])
                    )
                # Then max-pooling for the layer defined above
                maxp_layers[t].append(
                    MaxPool2D(pool_size=(self.max_len[t] - self.filter_size[i] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_layers[t][i])
                    )
              elif self.network_type == 'LSTM':
                maxp_layers[t].append(LSTM(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embed_layers[t]))
              elif self.network_type == 'RNN':
                maxp_layers[t].append(SimpleRNN(512, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)(embed_layers[t]))
              elif self.network_type == 'GRU':
                maxp_layers[t].append(GRU(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embed_layers[t]))
            inputs.append(input_layers[t])     

        if self.bert== True:   
            in_id = tf.keras.layers.Input(shape=(self.max_len['bert-word'],), name="input_ids")
            in_mask = tf.keras.layers.Input(shape=(self.max_len['bert-word'],), name="input_masks")
            in_segment = tf.keras.layers.Input(shape=(self.max_len['bert-word'],), name="segment_ids")
            bert_inputs = [in_id, in_mask, in_segment]
            inputs.append(in_id)
            inputs.append(in_mask)
            inputs.append(in_segment)
            bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
            reshape_layers['bert-word'] = tf.keras.layers.Reshape((self.max_len['bert-word'],
                                                 768, 1))(bert_output)
            maxp_layers['bert-word'] = []
            conv_layers['bert-word'] = []
            for i in range(len(self.filter_size)):
              # First the convolutional layer itself
              if self.network_type == 'CNN':
                  conv_layers['bert-word'].append(Conv2D(num_filters,
                      kernel_size=(self.filter_size[i], 768),
                      padding='valid', kernel_initializer='normal',
                      activation='relu')(reshape_layers['bert-word']))
                  # Then max-pooling for the layer defined above
                  maxp_layers['bert-word'].append(MaxPool2D(pool_size=(self.max_len['bert-word'] - self.filter_size[i] + 1,1),
                                                            strides=(1,1),
                                                            padding='valid')(conv_layers['bert-word'][i]))
              elif self.network_type == 'LSTM':
                maxp_layers['bert-word'].append(LSTM(512, 
                                                     dropout=0.2, 
                                                     recurrent_dropout=0.2, 
                                                     return_sequences=True)(bert_output))
              elif self.network_type == 'RNN':
                maxp_layers['bert-word'].append(SimpleRNN(512, dropout=0.2, 
                                                          recurrent_dropout=0.2,
                                                          return_sequences=True)(bert_output))
              elif self.network_type == 'GRU':
                maxp_layers['bert-word'].append(GRU(512, dropout=0.2, 
                                                    recurrent_dropout=0.2,
                                                    return_sequences=True)(bert_output))
        if self.skip_thoughts != False:
            self.max_len['skip_thoughts']=4800
            input_layers['skip_thoughts'] = Input(shape=(self.max_len['skip_thoughts'],),
                                    dtype='float32', name='skip_thoughts')
            # TODO: Why is this needed?
            reshape_layers['skip_thoughts'] = Reshape((self.max_len['skip_thoughts'],
                                         1, 1))(input_layers['skip_thoughts'])
                        # Now build three convolutional layers
                # First the convolutional layer itself
            conv_layers['skip_thoughts'] = []
            maxp_layers['skip_thoughts'] = []
            """conv_layers['skip_thoughts'].append(Conv2D(num_filters,
                       kernel_size=(1, 1),
                       padding='valid', kernel_initializer='normal',
                       activation='relu')( reshape_layers['skip_thoughts']))
                # Then max-pooling for the layer defined above
            maxp_layers['skip_thoughts'].append(MaxPool2D(pool_size=(self.max_len['skip_thoughts'], 1),
                          strides=(1, 1),
                          padding='valid')(conv_layers['skip_thoughts'][0]))"""
            hidden1 = Dense(512, activation='relu')(input_layers['skip_thoughts'])
            hidden2 = Dense(512,activation='relu')(hidden1)                
            inputs.append(input_layers['skip_thoughts'])     
        if self.glove != None:
            input_layers['glove_'+self.glove] = Input(shape=(self.max_len['glove_'+self.glove], 
                                                             self.glove_model.no_components),
                                    dtype='float32', name='glove_'+self.glove)
            aux_layer = tf.keras.layers.Reshape((self.max_len['glove_'+self.glove], 
                                                                           self.glove_model.no_components, 1))(input_layers['glove_'+self.glove])
            conv_layers['glove_'+self.glove] = []
            maxp_layers['glove_'+self.glove] = []
            for i in range(len(self.filter_size)):
              if self.network_type == 'CNN':
                conv_layers['glove_'+self.glove].append(
                    Conv2D(num_filters,
                           kernel_size=(self.filter_size[i], self.glove_model.no_components),
                           padding='valid', kernel_initializer='normal',
                           activation='relu')(aux_layer))
                maxp_layers['glove_'+self.glove].append(
                    MaxPool2D(pool_size=(self.max_len['glove_'+self.glove] - self.filter_size[i] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_layers['glove_'+self.glove][i]))
              elif self.network_type == 'LSTM':
               maxp_layers['glove_'+self.glove].append(LSTM(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(input_layers['glove_'+self.glove]))
              elif self.network_type == 'RNN':
                maxp_layers['glove_'+self.glove].append(SimpleRNN(512, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)(input_layers['glove_'+self.glove]))
              elif self.network_type == 'GRU':
                maxp_layers['glove_'+self.glove].append(GRU(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(input_layers['glove_'+self.glove]))
            inputs.append(input_layers['glove_'+self.glove])    
        drop_word= None
        tensors = [tensor for tt in maxp_layers for tensor in maxp_layers[tt]]
        if len(tensors) != 0:
            if len(tensors) == 1:
                concatenated_tensor = tensors[0]  
            elif len(tensors) > 1:
                concatenated_tensor = Concatenate(axis=1)([
                            tensor for tt in maxp_layers for tensor in maxp_layers[tt]])
            #Flatten the concatenated tensor
            flatten = Flatten()(concatenated_tensor)
            drop_word = Dropout(dropout)(flatten)
            # Set the output layer to a dense layer
            #word_embs = Dense(512, activation='softmax')(drop_word)  
            #drop_word = Dropout(dropout)(word_embs)
        # Flatten the concatenated tensor
        pv_embs = None
        if self.pv != False:
           input_layers['ParagraphVector'] = Input(shape=(self.pv_dim,),
                                    dtype='float32', name='ParagraphVector')
           inputs.append(input_layers['ParagraphVector'])
           pv_embs = Dense(512, activation='relu')(input_layers['ParagraphVector'])
           flatten = Flatten()(pv_embs)
           drop_pv = Dropout(dropout)(flatten)
           #IMPLEMENTAR CONCATENACION DE 1D VECTORS
           #concatenated_tensor_pv = Concatenate(axis=1)([drop,drop2])          
        if drop_word != None and pv_embs != None:
            concatenated_tensor = Concatenate(axis=-1)([drop_word, drop_pv])
        elif drop_word == None and pv_embs != None:
            concatenated_tensor = pv_embs
        else:
           concatenated_tensor = drop_word

        output = Dense(units=self.num_labels, activation='softmax')(concatenated_tensor)
        model = Model(inputs=inputs, outputs=output)
        optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        self.model = model
        self.model.summary()

    def set_bpe_tokenizer(self, tokenizer):
        """ This method sets the BPE tokenizer.

        Useful if an external model is provided

        :param tokenizer: a BPETokenizer instance
        """
        self.bpe_tokenizer = tokenizer

    def train_model(self, save_pref=None):
        """ This method does the actual training of the model.

        :param epochs: the number of epochs to perform. Defaults to 30
        :param batch_size: the batch size to use. Defaults to 50
        :param verbose: the degree of verbosity during training. Defaults to 1
        :param validation_split: the amount of data for validation. Defaults to 0.1
        :param save_pref: prefix to save the model weights. Defaults to None, in which
                          case the model weights are not stored
        """
        import random
        initialize_vars(sess)
        #if save_pref is not None:
        early_stopping_callback = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10,restore_best_weights=True)        
        save_model_callback = ModelCheckpoint('model.hdf5',
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
        callbacks = [early_stopping_callback, save_model_callback]
        self.history = self.model.fit(self.sequences, self.train_labels,
                       epochs=self.epochs, batch_size=self.batch_size,
                       verbose=self.trainig_verbose, callbacks=callbacks,
                       validation_data=(self.sequences_dev,self.dev_labels))
        self.best_epoch = early_stopping_callback.stopped_epoch - (early_stopping_callback.patience-1)
        pickle.dump(self.history.history,open('hist','wb'))
        self.model_status = 'TRAINED'

    def read_embeddings(self, embed_path):
        """ This method reads a pre-trained set of embeddings

        Returns the embedding matrix

        :param embed_path: the file where the embeddings are stored
        """
        self.logger.info("Reading embeddings from {}".format(embed_path))
        embedings_index = {}
        f = open(embed_path)
        for line in f:
            values = line.split()
            try:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                self.logger.warning('Warning! Embedding not imported: \n {}'.format(values))

            embedings_index[word] = coefs
        f.close()
        return embedings_index

    def filter_embeddings(self, embeddings, words):
        """ This method takes up a set of embeddings and filters it according to the
        words present in words.

        Returns the filtered set of embeddings.

        :param embeddings: the embedding matrix to be filtered
        :param words: a dictionary containing the words to be considered
        """
        self.logger.info("Filtering embeddings...")
        embed_dim = len(next(iter(embeddings.values())))
        filtered = np.zeros((len(words) + 1, embed_dim))
        for word, i in words.items():
            embed = embeddings.get(word)
            if embed is not None:
                filtered[i] = embed
        return filtered

    def save_model(self, model_file):
        """ This method saves the model parameters to a given file

        :param model_file: the file to save the parameters to
        """
        pickle.dump([self.network_type,
                     self.embed_dims,
                     self.pretrained_embeds,
                     self.max_len,
                     self.token_types,
                     self.bpe_tokenizer,
                     self.num_labels,
                     self.train_labels,
                     self.max_len,
                     self.vocabulary,
                     self.voc_size,
                     self.tokenizer_bert,
                     self.bert_path
                    ], open(model_file, 'wb'))
    

    def load_model(self, model_data):
        """ This method loads the model parameters from a given file

        :param model_data: the file to load the parameters from
        """
        [self.network_type,
         self.embed_dims,
         self.pretrained_embeds,
         self.max_len,
         self.token_types,
         self.bpe_tokenizer,
         self.num_labels,
         self.train_labels,
         self.max_len,
         self.vocabulary,
         self.voc_size,
         self.tokenizer_bert,
         self.bert_path] = pickle.load(open(model_data, 'rb'))



    def predict(self, test_path, nbest=1):
        """ This method takes up a data file, and launches prediction using the model.

        The model needs to have already been trained (or loaded).

        Returns a list of lists of dictionaries [[{'id': id, 'score':score}]]

        :param data: the csv file to read the data from
        :param separator: the separator for the csv file
        :param nbest: the number of nbest predictions to return
        :param sequence_field: what column the sequences are in the csv file. Defaults to 1
        :param label_field: what column the labels are in the csv file. Defaults to 2
        """
        self.set_test(test_path)
        
        prediction = self.model.predict(self.sequences_test)
        res = []
        for p in prediction:
            args = (-p).argsort()
            row = [{'id': str(a), 'score': str(p[a])} for a in args[0:nbest]]
            res.append(row)
        return prediction


    def generator(self, features, labels, batch_size):
        # Create empty arrays to contain batch of features and labels#
        #print(self.train_sequences.keys())
        while True:
            first=0
            last=first+batch_size
            batch_features = {}
            batch_labels = []
            while True:
                for tok_type in self.token_types:
                    batch_features[tok_type]= np.stack(self.train_sequences[tok_type][first:last], axis=0)
                    batch_labels = self.train_labels[first:last]
                first = last
                if last + batch_size < len(self.train_labels):
                    last += batch_size
                else:
                    break
                yield batch_features, batch_labels
    
    
    def confidence_interval(self, prediction):
    
        
        results = {}
        y_true = []
        for i in self.test_labels:
            y_true.append(np.argmax(i))
        
        y_pred=[]
        for i in prediction:
            y_pred.append(np.argmax(i))
            
        for i in range(1000):
            rand_pred = []
            rand_true = []
            for j in range (len(y_pred)):
                r = random.randint(0, len(prediction)-1)
                rand_pred.append(y_pred[r])
                rand_true.append(y_true[r])
            res = self.metrics(rand_true, rand_pred)    
            for k in res.keys():
                if k in results.keys():
                    results[k].append(res[k])
                else:
                    results[k]=[res[k]]
        
        for k in results.keys():
            print('%s mean: %.1f std: %.1f ' % (k, np.mean(results[k]), np.std(results[k])))
        return results
    
    def metrics(self, y_true, y_pred):
        if self.num_labels == 2:
            return {'f1': f1_score(y_true, y_pred)*100,
                    'accuracy': accuracy_score(y_true, y_pred)*100}
        else:
            return {'f1': f1_score(y_true, y_pred, average='macro')*100,
                    'accuracy': accuracy_score(y_true, y_pred)*100}
        
                    
