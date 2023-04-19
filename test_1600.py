import _pickle as pickle
import sys
import os
import sys
sys.path.append('./')
import torch
import torch.optim as optim
from torch.utils import data

# from FCN import FCNNet
from Net1600 import TransUNet as Net

from TU.utils import *
from TU.config import process_config
import pandas as pd
import time
from TU.data_generator import RNASSDataGenerator

from TU.data_generator import Dataset_test_1600
import collections

args = get_args()

from TU.postprocess import postprocess
def getbpseq_new(dict,true_seq,seq_name,seq_lens):
    for i in range(seq_lens):
        list1 = [j + 1 for j in range(seq_lens)]
        list2 = []
        for j in range(seq_lens):
            list2.append(dict[j+1])
        list3 = []
        for j in true_seq:
            list3.append(j)
        download = './seq/' + seq_name

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
def getcsv(seq_names,seq_lens_list,run_time,accuracy,recall,precision,f1_score):
    df = pd.DataFrame({'names':seq_names,'seq_lens':seq_lens_list,'run_times':run_time,'accuracy':accuracy,'recall':recall,'precision':precision,'f1':f1_score})
    df.to_csv("test1600.csv",index=False)
    return

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
    result_no_train = []
    batch_n = 0
    ct_dict_all = [] 
    seq_names = []
    seq = []
    seq_lens_list = []
    run_time = []
    accuracy = []
    recall = []
    precision = []
    f1_score = []
    error = []
    pos_weight = torch.Tensor([800]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:

        if batch_n%100==0:
            print('Batch number: ', batch_n)
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        seq_names.append(seq_name[0])
        seq_lens_list.append(seq_lens.item())
        tik = time.time()
        with torch.no_grad():
            pred_contacts = net(seq_embedding_batch)
        contact_masks = torch.zeros_like(pred_contacts)
        x = torch.zeros([1,seq_lens,seq_lens])
        x = pred_contacts[:,:seq_lens,:seq_lens]
        pred_contacts = x
        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5) ## 1.6
        map_no_train = (u_no_train > 0.5).float()
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)
        dict = getmatch(map_no_train[0],seq_lens)
        true_seq = get_seq(seq_ori[0])
        try:
            getbpseq_new(dict,true_seq,seq_name[0],seq_lens.item())
        except:
            error.append(seq_name)
            print(seq_name)

        seq.append(true_seq)
        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i],
                                                                    contacts_batch.cpu()[i],seq_lens), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp

        accuracy.append(result_no_train_tmp[0][0])
        recall.append(result_no_train_tmp[0][1])
        precision.append(result_no_train_tmp[0][2])
        f1_score.append(result_no_train_tmp[0][3])
        ct_dict_all.append(dict)

    nt_exact_a,nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    print('Average testing accuracy score with pure post-processing: ', np.average(nt_exact_a))
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))
    getcsv(seq_names,seq_lens_list,run_time,accuracy,recall,precision,f1_score)

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(0)

    config_file = args.config
    test_file = args.test_files
    
    config = process_config(config_file)
    #加载模型
    MODEL_SAVED = 'models/TransUFold_1600.pt'

    BATCH_SIZE = config.BATCH_SIZE
    # if gpu is to be used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    seed_torch()

    print('Loading test file: ',test_file)
    if test_file == 'RNAStralign' or test_file == 'ArchiveII':
        test_data = RNASSDataGenerator('data/', test_file+'.cPickle')
    else:
        test_data = RNASSDataGenerator('data/',test_file+'.cPickle')

    seq_len = test_data.data_y.shape[-2]  #512
    print('Max seq length ', seq_len)     #512

    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}

    test_set = Dataset_test_1600(test_data)
    test_generator = data.DataLoader(test_set, **params)

    
    net = Net(img_ch=16)

    print('==========Start Loading==========')
    net.load_state_dict(torch.load(MODEL_SAVED,map_location='cuda:0'))
    print('==========Finish Loading==========')

    net.to(device)
    model_eval_all_test(net,test_generator)
    

if __name__ == '__main__':

    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()