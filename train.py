import _pickle as pickle
import sys
import os
import torch
import torch.optim as optim
from torch.utils import data
import subprocess
import sys
sys.path.append('./')
from Net import TransUNet as Net
from TU.utils import *
from TU.config import process_config
from TU.data_generator import RNASSDataGenerator, Dataset
import collections

def train(net,train_merge_generator,epoches_first):
    epoch = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.Tensor([256]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    u_optimizer = optim.Adam(net.parameters())
    
    print('======let’s training=====')
    epoch_rec = []
    for epoch in range(400):
        net.train()
        running_loss = 0
        steps_done = 0
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_merge_generator:


            contacts_batch = torch.Tensor(contacts.float()).to(device)

            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            
            pred_contacts = net(seq_embedding_batch)
            contact_masks = torch.zeros_like(pred_contacts)
            #contact_masks=[aa.tolist() for aa in contact_masks]
            #print(contact_masks)
            contact_masks[0, :seq_lens[0], :seq_lens[0]] = 1

            contact_masks[1, :seq_lens[1], :seq_lens[1]] = 1
            # Compute loss
            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()
            steps_done=steps_done+1
            running_loss += loss_u.item()
            if(steps_done - 1) % 1000 == 0:
                print('Training log: epoch: {}, step: {}, loss: {}'.format(
                        epoch, steps_done-1, (running_loss/steps_done)))

        if epoch > 50:

            torch.save(net.state_dict(),  f'../TransUFold_{epoch}.pt')

def main():
    torch.cuda.set_device(0)
    args = get_args()
    config_file = args.config
    config = process_config(config_file)
    print('Here is the configuration of this run: ')
    print(config)

    os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
    BATCH_SIZE = config.BATCH_SIZE
    LOAD_MODEL = config.LOAD_MODEL

    epoches_first = 200
    train_files = args.train_files

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_torch()

    train_data_list = []
    for file_item in train_files:
        print('Loading dataset: ',file_item)
        train_data_list.append(RNASSDataGenerator('data/',file_item+'.cPickle'))
    print('Data Loading Done!!!')

    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True
             }

    train_merge = Dataset(train_data_list)
    train_merge_generator = data.DataLoader(train_merge, **params)
    
    net = Net(img_ch=16)
    net.to(device)

    train(net,train_merge_generator,epoches_first)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()
