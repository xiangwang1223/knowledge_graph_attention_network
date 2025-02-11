import torch
import numpy as np
import pandas as pd

from data_loader.loader_base import DataLoaderBase


class DataLoaderBPRMF(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.print_info(logging)


    def print_info(self, logging):
        logging.info('n_users:     %d' % self.n_users)
        logging.info('n_items:     %d' % self.n_items)
        logging.info('n_cf_train:  %d' % self.n_cf_train)
        logging.info('n_cf_test:   %d' % self.n_cf_test)


