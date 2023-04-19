
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data
sys.path.append('./')
import pickle as cPickle
from Net import TransUNet as NET
from Net1600 import TransUNet as NET1600
from TU.utils import *
from TU.config import process_config
import pdb
import time
from TU.data_generator import RNASSDataGenerator
from TU.data_generator import Dataset_test_1600
from TU.data_generator import RNASSDataGenerator, Dataset
from TU.data_generator import Dataset_test
from TU.postprocess import postprocess
import collections
import process_data_1600
import process_data_newdataset

import subprocess




def one_hot(seq):
    RNN_seq = seq
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
            [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[0] * len(BASES)]) for base
            in RNN_seq])

    return feat

def clean_pair(pair_list,seq):
    for item in pair_list:
        if seq[item[0]] == 'A' and seq[item[1]] == 'U':
            continue
        elif seq[item[0]] == 'C' and seq[item[1]] == 'G':
            continue
        elif seq[item[0]] == 'U' and seq[item[1]] == 'A':
            continue
        elif seq[item[0]] == 'G' and seq[item[1]] == 'C':
            continue
        else:
            print('%s+%s'%(seq[item[0]],seq[item[1]]))
            pair_list.remove(item)
    return pair_list
def getbpseq_new(dict,true_seq,seq_name,seq_lens):
    for i in range(seq_lens):
        list1 = [j + 1 for j in range(seq_lens)]
        list2 = []
        for j in range(seq_lens):
            list2.append(dict[j+1])
        list3 = []
        for j in true_seq:
            list3.append(j)
        download = './results/' + seq_name

        f = open(download, 'w')
        for k in range(0, seq_lens):
            f.write(str(list1[k]))
            f.write('   ')
            f.write(str(list3[k]))
            f.write('   ')
            f.write(str(list2[k]))
            f.write('\n')
        f.close()
def get_seq(seq_ori):
    seq = ''
    for i in seq_ori:
        for j,k in enumerate(i):
            if k == 1 and j == 0:
                seq += 'A'
            elif k == 1 and j ==1:
                seq += 'U'
            elif k == 1 and j ==2:
                seq += 'C'
            elif k == 1 and j ==3:
                seq += 'G'
            elif i.sum() == 0:
                seq += 'N'
                
    return seq
def getmatch(map_no_train,seq_lens):
    dict = {}
    for i in range(seq_lens):
        if (map_no_train[i].sum() ==1):
            for j in range(seq_lens):
                if(map_no_train[i][j] == 1):
                    dict[i + 1] = j + 1
        else:
            dict[i + 1] = 0
    return dict

def model_eval_all_test(net,test_generator):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.train()

    batch_n = 0
    ct_dict_all = [] 
    seq_names = []
    seq = []

    error = []
    pos_weight = torch.Tensor([256]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:

        if batch_n%10==0:
            print('Batch number: ', batch_n)
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        seq_names.append(seq_name[0])


        with torch.no_grad():
            pred_contacts = net(seq_embedding_batch)

        contact_masks = torch.zeros_like(pred_contacts)
        contact_masks[:, :seq_lens, :seq_lens] = 1

        x = torch.zeros([1,seq_lens,seq_lens])
        x = pred_contacts[:,:seq_lens,:seq_lens]
        pred_contacts = x
        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5) ## 1.6
        map_no_train = (u_no_train > 0.5).float()

        dict = getmatch(map_no_train[0],seq_lens)
        true_seq = get_seq(seq_ori[0])
        try:
            getbpseq_new(dict,true_seq,seq_name[0],seq_lens.item())
        except:
            error.append(seq_name)
            print(seq_name)

        seq.append(true_seq)


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(0)

    LOAD_MODEL = True

    MODEL_SAVED_512 = 'models/TransUFold_512.pt'
    MODEL_SAVED_1600 = 'models/TransUFold_1600.pt'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_torch()
        
    #####预测
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 6,
              'drop_last': True}
    if os.path.exists('./predict_cPickle/512.cPickle'):
        test_data_512 = RNASSDataGenerator('./predict_cPickle/', '512' + '.cPickle')
        test_set = Dataset_test(test_data_512)
        test_generator = data.DataLoader(test_set, **params)
        net = NET(img_ch=16)
        print('==========Start Loading Pretrained Model==========')
        net.load_state_dict(torch.load(MODEL_SAVED_512,map_location='cuda:0'))
        net.to(device)
        print('==========Finish Loading Pretrained Model==========')
        model_eval_all_test(net,test_generator)
        print('==========Done!!! Please check results folder for the 512 predictions!==========')
        torch.cuda.empty_cache()
        os.remove('./predict_cPickle/512.cPickle')
    if os.path.exists('./predict_cPickle/1600.cPickle'):
        test_data_1600 = RNASSDataGenerator('./predict_cPickle/', '1600' + '.cPickle')
        test_set = Dataset_test_1600(test_data_1600)
        test_generator = data.DataLoader(test_set, **params)
        net = NET1600(img_ch=16)
        print('==========Start Loading Pretrained Model==========')
        net.load_state_dict(torch.load(MODEL_SAVED_1600,map_location='cuda:0'))
        net.to(device)
        print('==========Finish Loading Pretrained Model==========')
        model_eval_all_test(net,test_generator)
        print('==========Done!!! Please check results folder for the 1600 predictions!==========')
        torch.cuda.empty_cache()
        os.remove('./predict_cPickle/1600.cPickle')

    
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    print('Welcome using TransUFold!')
    ###处理预测序列
    file_dir = sys.argv[1] + '/'  
    all_files = os.listdir(file_dir)
    all_files_list_512 = []
    all_files_list_1600 = []
    i = 0
    for index, item_file in enumerate(all_files):
        print(index)
        print(item_file)

        t0 = subprocess.getstatusoutput('awk \'NR==2{sub(/\\n/,""); print}\' ' + file_dir + item_file)

        seq = ''.join(str(t0[1][:-1]))

        try:
            one_hot_matrix = one_hot(seq.upper())
        except:
            pdb.set_trace()
        # 配对信息
        pair_dict_all_list = []
        seq_name = item_file
        seq_len = len(seq)

        if index % 1000 == 0:
            print('current processing %d/%d' % (index + 1, len(all_files)))
        if seq_len > 512 and seq_len <= 1600:
            ##cut all to 1600 length

            one_hot_matrix_1600 = np.zeros((1600, 4))
            one_hot_matrix_1600[:seq_len, ] = one_hot_matrix
            ss_label_1600 = np.zeros((1600, 3), dtype=int)


            sample_tmp = RNA_SS_data(seq=one_hot_matrix_1600, ss_label=ss_label_1600, length=seq_len, name=seq_name,
                                     pairs=pair_dict_all_list)
            all_files_list_1600.append(sample_tmp)
            i = i+1
        if seq_len > 0 and seq_len <= 512:

            ##cut all to 512 length

            one_hot_matrix_512 = np.zeros((512,4))
            one_hot_matrix_512[:seq_len,] = one_hot_matrix
            ss_label_512 = np.zeros((512,3),dtype=int)

            sample_tmp = RNA_SS_data(seq=one_hot_matrix_512,ss_label=ss_label_512,length=seq_len,name=seq_name,pairs=pair_dict_all_list)
            all_files_list_512.append(sample_tmp)
            i = i+1
        
    print(i)

    cPickle.dump(all_files_list_512, open("./predict_cPickle/512.cPickle", "wb"))
    cPickle.dump(all_files_list_1600, open("./predict_cPickle/1600.cPickle", "wb"))
    main()
    #########end




