import networkx as nx
import numpy as np
import random
random.seed(42)
np.random.seed(42)
from tqdm import tqdm
import logging


def stochastic_pruning(adj,p_del, p_add):
  print("Loading adjacency matrix..")
  print("Done! Calculating spectral gap...")
  print("Done!...")
  G = nx.from_scipy_sparse_matrix(adj)
  print(G)
  before_edges = G.number_of_edges()
  Lold = nx.normalized_laplacian_matrix(G)
  valsold,vecsold = np.linalg.eigh(Lold.A)
  candidates = random.sample(list(G.edges()), G.number_of_edges())
  candidates = np.array(list(set(map(tuple, candidates))))
  rem_edges = []
  add_edges = []
  print(f'Pruning with = {p_del, p_add}')
  for i in tqdm(np.nditer(np.arange(candidates.shape[0]))):
          proj = 1.0
          du = G.degree(candidates[i][0])
          dv = G.degree(candidates[i][1])
          fu = vecsold[1][candidates[i][0]]
          fv = vecsold[1][candidates[i][1]]
          cond1 = (proj**2)*valsold[1]+2*(1-valsold[1])
          cond11 = (np.sqrt(du+1)-np.sqrt(du))/np.sqrt(du+1)*(fu**2)
          cond12 = (np.sqrt(dv+1)-np.sqrt(dv))/np.sqrt(dv+1)*(fv**2)
          cond2 = (2*fu*fv)/np.sqrt(du+1)*np.sqrt(dv+1)
          final = cond1*(cond11+cond12) 
          if (final<cond2):  
              if np.random.random() < p_del:
                        np.random.choice(candidates[i])
                        #print("========================================================")
                        #print(f"Deleting edge {candidates[i][0], candidates[i][1]}")
                        #print("========================================================")
                        G.remove_edge(candidates[i][0],candidates[i][1])
                        rem_edges.append(candidates[i])
                        du = G.degree(candidates[i][0])
                        dv = G.degree(candidates[i][1]) 
                        Lnew = nx.normalized_laplacian_matrix(G)
                        delta_w = 1 - 2 * Lnew[candidates[:, 0], candidates[:, 1]].A1
                        delta_eigvals = valsold + delta_w[:, None] * ((vecsold[candidates[:, 0]] - vecsold[candidates[:, 1]])**2 
                                                                                    - valsold * (
                                                                    vecsold[candidates[:, 0]] ** 2 + vecsold[candidates[:, 1]] ** 2))
                        valsnew = np.sort(delta_eigvals)[0][1]
                        vecsnow = ((((vecsold[1][candidates[i][0]]).T*Lnew*(vecsold[1][candidates[i][1]]))
                                                  /(valsold[1]))*(vecsold[1][candidates[i][0]])).toarray()
                            
                        largest_eigenvector = vecsnow[:, np.argmax(valsnew)]
                        proj = np.dot(vecsnow[1], largest_eigenvector)/np.linalg.norm(largest_eigenvector)
                        valsold[1] = valsnew
                        fu=vecsnow[1][candidates[i][0]]
                        fv=vecsnow[1][candidates[i][1]]
          else:
            if np.random.random() < p_add:
                        np.random.choice(candidates[i])
                        #print("========================================================")
                        #print(f"Deleting edge {candidates[i][0], candidates[i][1]}")
                        #print("========================================================")
                        G.add_edge(candidates[i][0],candidates[i][1])
                        add_edges.append(candidates[i])
                        du = G.degree(candidates[i][0])
                        dv = G.degree(candidates[i][1]) 
                        Lnew = nx.normalized_laplacian_matrix(G)
                        delta_w = 1 + 2 * Lnew[candidates[:, 0], candidates[:, 1]].A1
                        delta_eigvals = valsold + delta_w[:, None] * ((vecsold[candidates[:, 0]] - vecsold[candidates[:, 1]])**2 
                                                                                    - valsold * (
                                                                    vecsold[candidates[:, 0]] ** 2 + vecsold[candidates[:, 1]] ** 2))
                        valsnew = np.sort(delta_eigvals)[0][1]
                        vecsnow = ((((vecsold[1][candidates[i][0]]).T*Lnew*(vecsold[1][candidates[i][1]]))
                                                  /(valsold[1]))*(vecsold[1][candidates[i][0]])).toarray()
                            
                        largest_eigenvector = vecsnow[:, np.argmax(valsnew)]
                        proj = np.dot(vecsnow[1], largest_eigenvector)/np.linalg.norm(largest_eigenvector)
                        valsold[1] = valsnew
                        fu=vecsnow[1][candidates[i][0]]
                        fv=vecsnow[1][candidates[i][1]]



  print("=======================================================")
  print(f"Number of edges pruned = {len(rem_edges)}")
  print(f"Number of edges added = {len(add_edges)}")
  final_edges = (len(add_edges) - len(rem_edges))
  print(f"Number of edges modified = {final_edges} ")
  print(G)
  sparsity = (((final_edges))/before_edges)*100
  print("Graph Sparsity: GraphSparsity:[{:.2f}%]".format((sparsity)))
  logging.info("Graph Sparsity: GraphSparsity:[{:.2f}%]".format((sparsity)))
 
  return nx.adjacency_matrix(G)