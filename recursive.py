import os
import random
import argparse

import torch
import torch.nn as nn
import numpy as np

import net as net
from utils import load_data, sparse_mx_to_torch_sparse_tensor
from sklearn.metrics import f1_score
import pdb
import pruning
import copy
from scipy.sparse import coo_matrix
from StochasticPruning import stochastic_pruning
from scipy.sparse import csgraph
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import logging


logger = logging.getLogger()
fhandler = logging.FileHandler(filename='experiments/New/NewPubmed/SparsePubmed0.1.log', mode='a')
formatter = logging.Formatter('%(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


pdel = 7
before_edges = 44327     #####Citeseer = 4676, Cora = 5278, Pubmed = 44327###

print(f"Pruning with ratio = {pdel}%")
logging.info(f'Pruning with Pdel - {pdel}%')
logging.info("")
print("Loading dataset...")
oldadj, features, labels, idx_train, idx_val, idx_test = load_data('pubmed')
node_num = features.size()[0]
class_num = labels.numpy().max() + 1
features = features.cuda()
labels = labels.cuda()
final_acc_holder = []
sparsity_holder = []
wei_sparsity = []
fig = plt.figure()
def run_fix_mask(args, seed, rewind_weight_mask):

          pruning.setup_seed(seed)
          loss_func = nn.CrossEntropyLoss()
          net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
          pruning.add_mask(net_gcn)
          net_gcn = net_gcn.cuda()
          net_gcn.load_state_dict(rewind_weight_mask)
          adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
          
          for name, param in net_gcn.named_parameters():
              if 'mask' in name:
                  param.requires_grad = False

          optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
          acc_test = 0.0
          best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}

          for epoch in range(200):

              optimizer.zero_grad()
              output = net_gcn(features, adj)
              loss = loss_func(output[idx_train], labels[idx_train])
              loss.backward()
              optimizer.step()
              with torch.no_grad():
                  output = net_gcn(features, adj, val_test=True)
                  acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
                  acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
                  if acc_val > best_val_acc['val_acc']:
                      best_val_acc['val_acc'] = acc_val
                      best_val_acc['test_acc'] = acc_test
                      best_val_acc['epoch'] = epoch
      
              print("(Fix Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                      .format(epoch, acc_val * 100, acc_test * 100, 
                                      best_val_acc['val_acc'] * 100, 
                                      best_val_acc['test_acc'] * 100, 
                                      best_val_acc['epoch']))

          return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc['epoch'], adj_spar, wei_spar


def run_get_mask(args, seed, imp_num, rewind_weight_mask=None):

          pruning.setup_seed(seed)

          loss_func = nn.CrossEntropyLoss()

          net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
          pruning.add_mask(net_gcn)
          net_gcn = net_gcn.cuda()

          if args['weight_dir']:
              print("load : {}".format(args['weight_dir']))
              encoder_weight = {}
              cl_ckpt = torch.load(args['weight_dir'], map_location='cuda')
              encoder_weight['weight_orig_weight'] = cl_ckpt['gcn.fc.weight']
              ori_state_dict = net_gcn.net_layer[0].state_dict()
              ori_state_dict.update(encoder_weight)
              net_gcn.net_layer[0].load_state_dict(ori_state_dict)

          if rewind_weight_mask:
              net_gcn.load_state_dict(rewind_weight_mask)
              if not args['rewind_soft_mask'] or args['init_soft_mask_type'] == 'all_one':
                  pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)
              adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
          else:
              pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)

          optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

          acc_test = 0.0
          best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}
          rewind_weight = copy.deepcopy(net_gcn.state_dict())
          for epoch in range(args['total_epoch']):
              
              optimizer.zero_grad()
              output = net_gcn(features, adj)
              loss = loss_func(output[idx_train], labels[idx_train])
              loss.backward()
              pruning.subgradient_update_mask(net_gcn, args) # l1 norm
              optimizer.step()
              with torch.no_grad():
                  output = net_gcn(features, adj, val_test=True)
                  acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
                  acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
                  if acc_val > best_val_acc['val_acc']:
                      best_val_acc['test_acc'] = acc_test
                      best_val_acc['val_acc'] = acc_val
                      best_val_acc['epoch'] = epoch
                      best_epoch_mask = pruning.get_final_mask_epoch(net_gcn, adj_percent=args['pruning_percent_adj'], 
                                                                              wei_percent=args['pruning_percent_wei'])

                  print("(Get Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                      .format(epoch, acc_val * 100, acc_test * 100, 
                                      best_val_acc['val_acc'] * 100,  
                                      best_val_acc['test_acc'] * 100,
                                      best_val_acc['epoch']))

          return best_epoch_mask, rewind_weight


def parser_loader():
          parser = argparse.ArgumentParser(description='GLT')
          ###### Unify pruning settings #######
          parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
          parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
          parser.add_argument('--total_epoch', type=int, default=300)
          parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
          parser.add_argument('--pruning_percent_adj', type=float, default=0.1)
          parser.add_argument('--weight_dir', type=str, default='')
          parser.add_argument('--rewind_soft_mask', action='store_true')
          parser.add_argument('--init_soft_mask_type', type=str, default='', help='all_one, kaiming, normal, uniform')
          ###### Others settings #######
          parser.add_argument('--dataset', type=str, default='citeseer')
          parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
          parser.add_argument('--lr', type=float, default=0.01)
          parser.add_argument('--weight-decay', type=float, default=5e-4)
          return parser


if __name__ == "__main__":

          parser = parser_loader()
          args = vars(parser.parse_args())
          print(args)
          seed_dict = {'cora': 2377, 'citeseer': 4428, 'pubmed': 3333} 
          seed = seed_dict[args['dataset']]
          rewind_weight = None


          for p in range(22):

              
              adj,rem_edges = stochastic_pruning(oldadj,pdel) 
              G = nx.from_scipy_sparse_matrix(adj)
              print("Number of edges pruned = ",len(rem_edges))
              print(G)
              sparsity = ((len(rem_edges))/before_edges)*100
              print("Graph Sparsity: GraphSparsity:[{:.2f}%]".format((sparsity)))
              with open("experiments/New/NewPubmed/InfoPubmed0.1.txt", "a") as f:
                            print(f"Number of edges pruned = {len(rem_edges)}", file=f)
                            print(G,file=f)
                            print("Graph Sparsity: GraphSparsity:[{:.2f}%]".format((sparsity)), file=f)
                            print(" ",file=f)
            
              oldadj = adj  
              adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
              adj = adj.cuda()
              
              final_mask_dict, rewind_weight = run_get_mask(args, seed, p, rewind_weight)
              rewind_weight['adj_mask1_train'] = final_mask_dict['adj_mask']
              rewind_weight['adj_mask2_fixed'] = final_mask_dict['adj_mask']
              rewind_weight['net_layer.0.weight_mask_train'] = final_mask_dict['weight1_mask']
              rewind_weight['net_layer.0.weight_mask_fixed'] = final_mask_dict['weight1_mask']
              rewind_weight['net_layer.1.weight_mask_train'] = final_mask_dict['weight2_mask']
              rewind_weight['net_layer.1.weight_mask_fixed'] = final_mask_dict['weight2_mask']

              best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(args, seed, rewind_weight)
              
    
                        
              #final_acc_holder.append(final_acc_test*100)
              #sparsity_holder.append(sparsity)
              #logging.disable()
              #sns.set()
              #plt.plot(sparsity_holder,final_acc_holder)
              #plt.xlabel('Graph Sparsity')
              #plt.ylabel('Test Accuracy')
              #plt.title('Braess-LTH-Citeseer-0.1')
              #logging.disable(logging.DEBUG)

      
              print("=" * 120)
              print("syd : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}]] Adj_Spar:[{:.2f}%]] Wei:[{:.2f}%]"
                  .format(p + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100,sparsity, wei_spar))
 
              logging.info(("syd : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj_Spar:[{:.2f}%]] Wei:[{:.2f}%]"
                  .format(p + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100, sparsity,wei_spar)))
              print("=" * 120)

# plt.savefig('experiments/Pubmed/PubmedRecursive0.1.png', dpi=300, bbox_inches='tight')
# plt.show()
logging.info("===============================================================================")
logging.info("")


