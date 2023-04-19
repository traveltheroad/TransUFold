import numpy as np
import os
import _pickle as cPickle
import collections
from TU.utils import *
from multiprocessing import Pool
from torch.utils import data
from collections import Counter
from random import shuffle
import torch
from itertools import permutations, product
import pdb

import math

perm = list(product(np.arange(4), np.arange(4)))


        
class RNASSDataGenerator(object):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.load_data()
        self.batch_pointer = 0

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        RNA_SS_data = collections.namedtuple('RNA_SS_data',
            'seq ss_label length name pairs')
        with open(os.path.join(data_dir, '%s' % self.split), 'rb') as f:
            self.data = cPickle.load(f,encoding='iso-8859-1')
        self.data_x = np.array([instance[0] for instance in self.data])
        self.data_y = np.array([instance[1] for instance in self.data])
        self.pairs = np.array([instance[-1] for instance in self.data])
        self.seq_length = np.array([instance[2] for instance in self.data])
        self.len = len(self.data)
        self.seq = list(p.map(encoding2seq, self.data_x))
        self.seq_max_len = len(self.data_x[0])
        self.data_name = np.array([instance[3] for instance in self.data])

    def next_batch(self, batch_size):
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        batch_x = self.data_x[bp:bp + batch_size]
        batch_y = self.data_y[bp:bp + batch_size]
        batch_seq_len = self.seq_length[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0

        yield batch_x, batch_y, batch_seq_len

    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact

    def next_batch_SL(self, batch_size):
        p = Pool()
        bp = self.batch_pointer
        data_y = self.data_y[bp:bp + batch_size]
        data_seq = self.data_x[bp:bp + batch_size]
        data_pairs = self.pairs[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0
        contact = np.array(list(map(self.pairs2map, data_pairs)))
        matrix_rep = np.zeros(contact.shape)
        yield contact, data_seq, matrix_rep

    def get_one_sample(self, index):
        data_y = self.data_y[index]
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        data_pair = self.pairs[index]
        data_name = self.data_name[index]

        contact= self.pairs2map(data_pair)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len, data_name


    def random_sample(self, size=1):
        # random sample one RNA
        # return RNA sequence and the ground truth contact map
        index = np.random.randint(self.len, size=size)
        data = list(np.array(self.data)[index])
        data_seq = [instance[0] for instance in data]
        data_stru_prob = [instance[1] for instance in data]
        data_pair = [instance[-1] for instance in data]
        seq = list(map(encoding2seq, data_seq))
        contact = list(map(self.pairs2map, data_pair))
        return contact, seq, data_seq


class RNASSDataGenerator_input(object):
    def __init__(self,data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.load_data()

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        RNA_SS_data = collections.namedtuple('RNA_SS_data',
                    'seq ss_label length name pairs')
        input_file = open(os.path.join(data_dir, '%s.txt' % self.split),'r').readlines()
        self.data_name = np.array([itm.strip()[1:] for itm in input_file if itm.startswith('>')])
        self.seq = [itm.strip().upper().replace('T','U') for itm in input_file if itm.upper().startswith(('A','U','C','G','T'))]
        self.len = len(self.seq)
        self.seq_length = np.array([len(item) for item in self.seq])
        self.data_x = np.array([self.one_hot_512(item) for item in self.seq])
        self.seq_max_len = 512
        self.data_y = self.data_x

    def one_hot_512(self,seq_item):
        RNN_seq = seq_item
        BASES = 'AUCG'
        bases = np.array([base for base in BASES])
        feat = np.concatenate(
                [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
                in RNN_seq])
        if len(seq_item) <= 512:
            one_hot_matrix_512 = np.zeros((512,4))
        else:
            one_hot_matrix_512 = np.zeros((512,4))
        one_hot_matrix_512[:len(seq_item),] = feat
        return one_hot_matrix_512

    def get_one_sample(self, index):

        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        #data_y = self.data_y[index]
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        data_name = self.data_name[index]

        return data_seq, data_len, data_name

# using torch data loader to parallel and speed up the data load process

class Dataset_Cut_concat_new(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data_seq, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len,80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8,l,l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            #contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            #contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])

        data_fcn_2 = np.concatenate((data_fcn,data_fcn_1),axis=0)
        return data_fcn_2, data_len, data_seq[:l], data_name


class Dataset_1600(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_list):
        'Initialization'
        self.data = data_list[0]

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        data_fcn = np.zeros((16, 1600, 1600))
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
            # data_fcn =  16 , l ,l
        return contact, data_fcn, matrix_rep, data_len, data_seq, data_name
    
        
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_list):
        'Initialization'
        self.data = data_list[0]

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        data_fcn = np.zeros((16, 512, 512))
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
            # data_fcn =  16 , l ,l
        return contact, data_fcn, matrix_rep, data_len, data_seq, data_name


class Dataset_test(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        data_fcn = np.zeros((16, 512, 512))

        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))

        return contact, data_fcn, matrix_rep, data_len, data_seq[:data_len], data_name
    
class Dataset_test_1600(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        data_fcn = np.zeros((16, 1600, 1600))

        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))

        return contact, data_fcn, matrix_rep, data_len, data_seq[:data_len], data_name

class Dataset_Cut_concat_new_merge_two(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data1, data2):
        'Initialization'
        self.data1 = data1
        self.data2 = data2
        self.merge_data()
        self.data = self.data2

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def merge_data(self):
        self.data2.data_x = np.concatenate((self.data1.data_x[:,:600,:],self.data2.data_x),axis=0)
        self.data2.data_y = np.concatenate((self.data1.data_y[:,:600,:],self.data2.data_y),axis=0)
        self.data2.seq_length = np.concatenate((self.data1.seq_length,self.data2.seq_length),axis=0)
        self.data2.pairs = np.concatenate((self.data1.pairs,self.data2.pairs),axis=0)
        self.data2.data_name = np.concatenate((self.data1.data_name,self.data2.data_name),axis=0)
        self.data2.len = len(self.data2.data_name)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len,80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8,l,l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])
        zero_mask = z_mask(data_len)[None, :, :, None]
        label_mask = l_mask(data_seq, data_len)
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        feature[:,:data_len,:data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape((-1,data_len,data_len))
        feature = np.concatenate((data_fcn,feature), axis=0)

        data_fcn_2 = np.concatenate((data_fcn,data_fcn_1), axis=0)
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name



def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l

