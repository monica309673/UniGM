import argparse
from pyrsistent import freeze
from loader import MoleculeDataset
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import GNN, GNN_graphpred
from backbone import GNN_fuse, GNN_graphpred_fuse 
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split, random_split
import pandas as pd
import os
import shutil
import time

criterion = nn.BCEWithLogitsLoss(reduction = "none")
def train(args, model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))   
        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()

def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]

def load_pretrained_model(name, model_path='saved_model'):
    pkl = torch.load(os.path.join(model_path, name+'.pth'), )
    state_dict = {}
    if name == 'simgrace':
        for k, v in pkl.items():
            state_dict[k] = v

    if name == 'graphlog':
        for k, v in pkl.items():
            state_dict[k] = v
    elif name == 'graphmvp6':
        for k, v in pkl.items():
            state_dict[k] = v 
    elif name == 'motif_pretrain': 
        for k, v in pkl.items():
            state_dict[k] = v    
    elif name == 'graphcl_80': 
        for k, v in pkl.items():
            state_dict[k] = v   
    elif name == 'supervised_masking': 
        for k, v in pkl.items():
            state_dict[k] = v  
    elif name == 'supervised_edgepred': 
        for k, v in pkl.items():
            state_dict[k] = v  
    
    return state_dict
            
def build_model(args, num_tasks, rnn, model_dict=None):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = GNN_graphpred_fuse(args.num_layer, args.emb_dim, args.freeze, rnn, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type).to(device)

    if model_dict is None:
        state_dict_graphlog = load_pretrained_model('graphlog')  
        state_dict_graphmvp = load_pretrained_model('graphmvp6')
        state_dict_motif_pretrain = load_pretrained_model('motif_pretrain')
        state_dict_graphcl = load_pretrained_model('graphcl_80') 
    else:
        state_dict_graphlog = model_dict['graphlog'] 
        state_dict_graphmvp = model_dict['graphmvp6']
        state_dict_motif_pretrain = model_dict['motif_pretrain']
        state_dict_graphcl = model_dict['graphcl_80']

    state_dict_new = model.state_dict()
    for name in state_dict_graphcl:  
        w1 = state_dict_graphlog[name].to(device)
        w2 = state_dict_graphmvp[name].to(device)
        w3 = state_dict_motif_pretrain[name].to(device)
        w4 = state_dict_graphcl[name].to(device)
        wt = state_dict_new['gnn.' + name]

        if wt.size() == w1.size():
            state_dict_new['gnn.' + name] = (w1 + w2 + w3 + w4) / 4.0
        else:
            state_dict_new['gnn.' + name] = torch.cat((w1.unsqueeze(0), w2.unsqueeze(0), w3.unsqueeze(0), w4.unsqueeze(0)), dim=0)
        if 'mean' in name:
            state_dict_new['gnn.' + name] = w1 * 0.0
        elif 'var' in name:
            state_dict_new['gnn.' + name] = w1 * 0.0 + 1.0

    model.load_state_dict(state_dict_new)
    return model

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=2,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'sider', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--freeze', type=int, default = 1, help='whether to freeze the pre-trained models.')  
    parser.add_argument('--RNN_hid', type=int, default = 8, help='Hidden dimension of RNN.')
    parser.add_argument('--RNN_layer', type=int, default = 2, help='The num of layers of RNN.')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=42, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("./chem/dataset/" + args.dataset, dataset=args.dataset)
    if args.split == "scaffold":
        smiles_list = pd.read_csv('./chem/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    rnn = nn.LSTM(input_size = 1, hidden_size = args.RNN_hid, num_layers = args.RNN_layer, batch_first=True).to(device)
    fused_model = build_model(args, num_tasks, rnn)
    num_params = sum(param.numel() for param in fused_model.parameters())
    print('num_params:', num_params)
    model_param_group = []
    model_param_group.append({"params": fused_model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": fused_model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": fused_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": fused_model.rnn.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")

    train_time = 0
    test_time = 0
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))    
        start_train = time.time()    
        train(args, fused_model, device, train_loader, optimizer)
        end_train = time.time()
        train_time = train_time + end_train - start_train
        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, fused_model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, fused_model, device, val_loader)
        start_test = time.time()
        test_acc = eval(args, fused_model, device, test_loader)
        end_test = time.time()
        test_time = test_time + end_test - start_test

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)    
        print('Memory Cost:', torch.cuda.max_memory_allocated())
    
    end_time = time.time()
    cost_time = end_time - start_time
    print('best epoch: ', val_acc_list.index(max(val_acc_list)))
    print('best auc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])
    exp_path = os.getcwd() + '/results/{}/'.format(args.dataset)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    df = pd.DataFrame({'train':train_acc_list,'valid':val_acc_list,'test':test_acc_list})
    df.to_csv(exp_path + '{}_seed{}_align.csv'.format(str(args.freeze), args.runseed))

    logs = 'Dataset:{},Freeze:{},Train Time:{}, Test Time:{}, Seed:{}, Best Epoch:{}, Best Acc:{:.5f}'.format(args.dataset, str(args.freeze), train_time, test_time, args.runseed, val_acc_list.index(max(val_acc_list)), test_acc_list[val_acc_list.index(max(val_acc_list))])
    with open(exp_path + '{}_log.csv'.format(args.dataset),'a+') as f:
        f.write('\n')
        f.write(logs)

    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
