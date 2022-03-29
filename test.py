#!/usr/bin/env python3 -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 5 15:31:15 2018

Author: Zuzanna Parcheta <zparcheta@sciling.com>
        Germán Sanchis-Trilles <gsanchis@sciling.com>

(c) 2018 Sciling, SL
"""
import yaml
from seqclass import SeqClass
from bpetokenizer import BPETokenizer
import argparse
import numpy as np

plot = False
try:
    from tensorflow.keras.utils import plot_model
    plot = True
except:
    plot = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clasiffier for gastrofy.')
    parser.add_argument('--data', help='path to data file')
    # Options for establishing what type of tokens to use
    parser.add_argument('--emb_size_word', default=0, type=int,
                         help='Size of word embedding')
    parser.add_argument('--emb_size_char', default=0, type=int,
                         help='Size of char embedding')
    parser.add_argument('--emb_size_bpe', default=0, type=int,
                         help='Size of bpe embedding')
    # Options for loading pre-trained embeddings
    parser.add_argument('--pretrained_emb_word', default=None,
                         help='path to pretrained word embedding')
    parser.add_argument('--pretrained_emb_char', default=None,
                         help='path to pretrained char embedding')
    parser.add_argument('--pretrained_emb_bpe', default=None,
                         help='path to pretrained bpe embedding')
    parser.add_argument('--filter_sizes', default='3',
                         help='filter sizes')
    parser.add_argument('--network', default='CNN',
                         help='network type')
    # Option for loading a pre-trained BPE model
    parser.add_argument('--bpe_model', default=None,
                         help='bpe model path (default: None)')
    # Options for establishing hwo to train
    parser.add_argument('--batch_size', default=50, type=int,
                         help='Batch size for training')
    # In case a pre-trained char embedding is used, what was the separator?
    parser.add_argument('--separator', default='A',
                         help='When embedding trained with chacarters, \
                         the separator used.')
    # Options to save and load the model
    parser.add_argument('--save_pref', default=None,
                         help='Output path to save model weights.')
    parser.add_argument('--save', default=None, help='File to save the model to')
    parser.add_argument('--load', default=None, help='File to load the model from')
    # Options to predict given a test set
    parser.add_argument('--predict', default=None, help='File to read test data')
    parser.add_argument('--nbest', default=1, type=int, help='Number of nbest for prediction')
    parser.add_argument('--dev', default=None, help='dev')
    parser.add_argument('--bert', default=False, help='t/f')
    parser.add_argument('--glove', default=None, help='word, bpe, char, None')
    parser.add_argument('--pv', default=False, help='t/f')
    parser.add_argument('--skip_thoughts', default=False, help='path to model')
    parser.add_argument('--weights', default=None, help='t/f')
    parser.add_argument('--delimiter', default='\t', help='deparator')
    args = parser.parse_args()


    stream = open('autoconfig.yml', 'r')
    params = yaml.load_all(stream, Loader=yaml.FullLoader)
    #  --skip_thoughts skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424
    filters = list(map(int, args.filter_sizes.split(',')))
    if args.data is not None:
        m = SeqClass(embed_dims={'word': args.emb_size_word,
                                 'bpe': args.emb_size_bpe,
                                 'char': args.emb_size_char},
                     pretrained_embeds={'word': args.pretrained_emb_word,
                                        'bpe': args.pretrained_emb_bpe,
                                        'char': args.pretrained_emb_char},
                     white_space_char=args.separator, delimiter=args.delimiter)
        m.filter_size = filters
        m.network_type = args.network
        m.skip_thoughts = args.skip_thoughts

        #"""
        if args.bert == 'True' or args.bert == 'true' :        # is true
            m.bert = True
        else:
            m.bert = False
        if args.pv == 'True' or args.pv == 'true':
            m.pv = True
        else:
            m.pv = False
        m.glove = args.glove

        if args.bpe_model is not None:
            tokenizer = BPETokenizer()
            tokenizer.load_model(args.bpe_model)
            m.set_bpe_tokenizer(tokenizer)

        if args.load != None:
            m.load_model(args.load)
        else:
            m.set_train(args.data)

            m.set_dev(args.dev)
        m.build_model()
        if plot == True:
            try:
                plot_model(m.model, to_file='model.png')
            except:
                pass
        if args.load != None:
            m.model.load_weights(args.weights)
        if args.load == None:
            if args.save is not None:
                m.save_model(args.save)

            m.train_model(save_pref=args.save_pref)

    if args.predict is not None:
        #m = SeqClass()
        #m.set_train(args.data, ';')
        #m.load_model(args.load)
        if args.predict =='datasets/ger/ger.test':
            for i in ['datasets/ger/ger.test2', args.predict]:
                print(i)
                res = m.predict(i)# 2566)#
                results =  m.confidence_interval(res)
                latex_string = '& ' + str(m.best_epoch) +' & '
                for k in results.keys():
                    latex_string += '%.1f$\\pm$%.1f & ' % (np.mean(results[k]), np.std(results[k]))
                print('***>>>' ,m.token_types, m.glove, m.bert, m.pv, m.filter_size, m.network_type, latex_string[:-2]+"\\\\")
        else:
            for i in [args.predict]:
                print(i)
                res = m.predict(i)# 2566)#

                if i == args.predict:
                    results =  m.confidence_interval(res)
                    latex_string = '& ' + str(m.best_epoch) +' & '
                    for k in results.keys():
                        latex_string += '%.1f$\\pm$%.1f & ' % (np.mean(results[k]), np.std(results[k]))
                    print('***>>>' ,m.token_types, m.glove, m.bert, m.pv, m.filter_size, m.network_type, latex_string[:-2]+"\\\\")
                else:
                    m.confidence_interval(res)

