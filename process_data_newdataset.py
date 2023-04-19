import pdb
import numpy as np
import os
import subprocess
import collections
import pickle as cPickle
import random
import sys

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

if __name__=='__main__':
    #读入文件夹
    file_dir = sys.argv[1]+'/'

    all_files = os.listdir(file_dir)
    random.seed(4)
    random.shuffle(all_files)
    
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    all_files_list = []

    for index,item_file in enumerate(all_files):

        t0 = subprocess.getstatusoutput('awk \'{print $2}\' '+file_dir+item_file)

        seq = ''.join(t0[1].split('\n'))
        #转换成one_hot
        if t0[0] == 0:
            try:
                one_hot_matrix = one_hot(seq.upper())
            except:
                pdb.set_trace()
        #配对信息
        t1 = subprocess.getstatusoutput('awk \'{print $1}\' '+file_dir+item_file)
        t2 = subprocess.getstatusoutput('awk \'{print $3}\' '+file_dir+item_file)
        if t1[0] == 0 and t2[0] == 0:
            pair_dict_all_list = [[int(item_tmp)-1,int(t2[1].split('\n')[index_tmp])-1] for index_tmp,item_tmp in enumerate(t1[1].split('\n')) if int(t2[1].split('\n')[index_tmp]) != 0]
        else:
            pair_dict_all_list = []
        seq_name = item_file
        seq_len = len(seq)
        pair_dict_all = dict([item for item in pair_dict_all_list if item[0]<item[1]])

        if index%100==0:
            print('current processing %d/%d'%(index+1,len(all_files)))
        if seq_len > 0 and seq_len <= 512:
            ss_label = np.zeros((seq_len,3),dtype=int)
            ss_label[[*pair_dict_all.keys()],] = [0,1,0]
            ss_label[[*pair_dict_all.values()],] = [0,0,1]
            ss_label[np.where(np.sum(ss_label,axis=1) <= 0)[0],] = [1,0,0]

            ##cut all to 512 length

            one_hot_matrix_512 = np.zeros((512,4))
            one_hot_matrix_512[:seq_len,] = one_hot_matrix
            ss_label_512 = np.zeros((512,3),dtype=int)
            ss_label_512[:seq_len,] = ss_label
            ss_label_512[np.where(np.sum(ss_label_512,axis=1) <= 0)[0],] = [1,0,0]

            sample_tmp = RNA_SS_data(seq=one_hot_matrix_512,ss_label=ss_label_512,length=seq_len,name=seq_name,pairs=pair_dict_all_list)
            all_files_list.append(sample_tmp)

        
    print(len(all_files_list))

    cPickle.dump(all_files_list,open("./data/Test.cPickle","wb"))
