#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 5 15:31:15 2018

Author: Zuzanna Parcheta <zparcheta@sciling.com>
        Germán Sanchis-Trilles <gsanchis@sciling.com>

(c) 2018 Sciling, SL
"""

import logging
from bpe import Encoder

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(name)s:%(levelname)s:%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')

class BPETokenizer():
    """ Wrapper class for calling the BPE tokenizer in the bpe module
    """
    def set_train(self, data_file):
        """ Set training data for the BPE tokenizer.

        :param data_file: The file with the data.
        """
        self.data = list(open(data_file).readlines())

    def set_data(self, data):
        """ Set the training data via a list of strings.

        :param data: The data matrix.
        """
        self.data = data

    def train_model(self, iterations=1000, pct_bpe=0.9):
        """ Train the BPE model.

        :param iterations: The number of iterations to perform.
        :param pct_bpe: The percentage of splits to perform.
        """
        self.encoder = Encoder(iterations, pct_bpe=pct_bpe)
        self.encoder.fit([x.lower() for x in self.data])

    def tokenize(self, data):
        """ Tokenize new data with a trained model.

        :param data: The list of strings to tokenize.
        """
        return self.encoder.tokenize(data)

    def save_model(self, model_file):
        """ Save the BPE model to a file.

        :param model_file: The file to save the model to.
        """
        logger.info("Saving BPE model to {}".format(model_file))
        import pickle
        pickle.dump(self.encoder, open(model_file, 'wb'))

    def load_model(self, model_file):
        """ Load the BPE model from a file.

        :param model_file: The file to load the model from.
        """
        logger.info("Loading BPE model from {}".format(model_file))
        import pickle
        self.encoder = pickle.load(open(model_file, 'rb'))


if __name__ == "__main__":
    import argparse
    description = "Build a BPE model, or perform BPE on a dataset"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data', help='File to load the data from')
    parser.add_argument('--save', help='File to store the model to')
    parser.add_argument('--load', help='File to load the model from')
    parser.add_argument('--test', help='file to run BPE on')
    args = parser.parse_args()
    if args.data:
        m = BPETokenizer()
        m.set_train(args.data)
        m.train_model()
        if args.save:
            m.save_model(args.save)
    if args.load:
        m = BPETokenizer()
        m.load_model(args.load)
    if args.test:
        test_data = open(args.test).readlines()
        [print(" ".join(m.tokenize(x.lower()))) for x in test_data]
