import networkx as nx
import numpy as np
import random
random.seed(4258031807)
np.random.seed(4258031807)
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
rem_edges = []
edge_holder = []

def stochastic_pruning(adj,pdel):
  print("Loading adjacency matrix..")
  G = nx.from_scipy_sparse_matrix(adj)
  print(G)   
  to_remove = random.sample(G.edges(),k=1)
  G.remove_edges_from(to_remove)
  before_edges = G.number_of_edges() 
  print("Done! Calculating spectral gap...")
  Lold = nx.normalized_laplacian_matrix(G)
  valsold,vecsold = np.linalg.eigh(Lold.A)
  print("Done!...")
  print("Calculating projection...")
  largest_eigenvector = vecsold[:, np.argmax(valsold)]
  proj = np.dot(vecsold[1], largest_eigenvector) / (np.linalg.norm(largest_eigenvector))
  print("Done!")
    
    
  samples = np.int((pdel/100)*before_edges)
  print(f"Sample size = {samples}")
  sampling = random.sample(list(G.edges()),samples)
  candidates = np.array(list(set(map(tuple, sampling))))
  print("Checking for Braess condition...")
    
    
  for i in tqdm(np.nditer(np.arange(candidates.shape[0]))):
          #proj = -0.006157414974414208
          du = G.degree(candidates[i][0])
          dv = G.degree(candidates[i][1])
          fu = vecsold[1][candidates[i][0]]
          fv = vecsold[1][candidates[i][1]]
          cond1 = (proj**2)*valsold[1]+2*(1-valsold[1])
          cond11 = np.divide(np.subtract(np.sqrt(du+1), np.sqrt(du)), np.sqrt(du+1)) * np.square(fu)
          cond12 = np.divide(np.subtract(np.sqrt(dv+1), np.sqrt(dv)), np.sqrt(dv+1)) * np.square(fv)
          #cond11 = (np.sqrt(du+1)-np.sqrt(du))/np.sqrt(du+1)*(fu**2)
          #cond12 = (np.sqrt(dv+1)-np.sqrt(dv))/np.sqrt(dv+1)*(fv**2)    

          cond2 = (2*fu*fv)/np.sqrt(du+1)*np.sqrt(dv+1)
          final = cond1*(cond11+cond12) 
          if (final<cond2): 
              if np.random.random() < pdel:
                        np.random.choice(candidates[i],replace=False)
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

#   print("=======================================================")
#   print("Number of edges pruned = ",len(rem_edges))
#   edge_holder.append(len(rem_edges))
#   edges = 5278
#   sparsity = ((len(rem_edges))/edges)*100
#   print(G)
#   print("Graph Sparsity: GraphSparsity:[{:.2f}%]".format((sparsity)))

#   with open("experiments/New/Coraa%.txt", "a") as f:
#             print(f"Number of edges pruned = {len(rem_edges)}", file=f)
#             print(G,file=f)
#             print(" ",file=f)
    
  #isolated_nodes = list(nx.isolates(G))
  #G.remove_nodes_from(isolated_nodes)
  return nx.adjacency_matrix(G),rem_edges



