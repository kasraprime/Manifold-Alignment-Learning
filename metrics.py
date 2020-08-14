## Code by Andre Nguyen: github.com/andrenguyen

import copy
import numpy as np
import scipy as sp
import random
import umap
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def knn(a,b,y,k=5,cosine=False):
    if cosine:
        neigh = KNeighborsClassifier(n_neighbors=k,metric='cosine')
    else:
        neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(b, y) 
    preds = neigh.predict(a)
    return accuracy_score(y, preds)


def rank_from(a,b,cosine=False):
    # To compare different embedding methods, regular distances are not meaningful.
    if cosine:
        M = 1-cosine_similarity(a,b)
    else:
        M = euclidean_distances(a,b)
    for row in range(len(M)):
        M[row] = sp.stats.rankdata(M[row], method='ordinal')
    return np.diag(M)


def rank_from_class(a,b,y,cosine=False):
    res = []
    if cosine:
        M = 1-cosine_similarity(a,b)
    else:
        M = euclidean_distances(a,b)
    for row in range(len(M)):
        idxs = [i for i in range(len(M)) if (y[i] == y[row])]
        ranks = sp.stats.rankdata(M[row], method='ordinal')
        res.append(min([ranks[i] for i in idxs]))
    return res


def object_identification_task_rank(a,b,y,cosine=False):  
    # Based on ranks.
    res = []
    if cosine:
        M = 1-cosine_similarity(a,b)
    else:
        M = euclidean_distances(a,b) 
    for row in range(len(M)):
        sorted_class = [e for _, e in sorted(zip(M[row],copy.deepcopy(y)))]
        sorted_match = [(e == y[row]) for i,e in enumerate(sorted_class)] 
        res.append(sorted_match)
    return res


def object_identification_task_classifier(a,b,y,a_train,b_train,cosine=False,lamb_std=1.0):  
    # Based on distances. 
    y_true = []
    y_pred = []
    y_dist = []
    if cosine:
        M_train = 1-cosine_similarity(a_train,b_train)
        M = 1-cosine_similarity(a,b)
    else:
        M_train = euclidean_distances(a_train,b_train) 
        M = euclidean_distances(a,b) 
    thres = np.mean(np.diag(M_train)) + lamb_std*np.std(np.diag(M_train))        
 
    for row in range(len(M)):
        y_dist.append([e for e in M[row]])
        y_pred.append([e <= thres for e in M[row]])
        y_true.append([label == y[row] for label in y])
    return y_true, y_pred, y_dist


def mean_reciprocal_rank(a,b,y,cosine=False):
    ranks = rank_from_class(a,b,y,cosine=cosine)
    reciprocal_ranks = [1/e for e in ranks] 
    return np.mean(reciprocal_ranks)


def corr_between(a,b,cosine=False):
    # Pick a pair, distance in A and distance in B should be correlated.
    a_dists = []
    b_dists = []
    for _ in range(10000):
        i1 = random.randrange(len(a))
        i2 = random.randrange(len(a))
        if cosine:
            a_dists.append(sp.spatial.distance.cosine(a[i1], a[i2]))
            b_dists.append(sp.spatial.distance.cosine(b[i1], b[i2]))
        else:
            a_dists.append(sp.spatial.distance.euclidean(a[i1], a[i2]))
            b_dists.append(sp.spatial.distance.euclidean(b[i1], b[i2])) 
    return sp.stats.pearsonr(a_dists,b_dists)[0]


def umap(a,b,cosine=False):
    ab = np.concatenate((a,b),axis=0)
    if cosine:
        embedding = umap.UMAP(n_components=2, metric='cosine').fit_transform(ab)
    else:
        embedding = umap.UMAP(n_components=2).fit_transform(ab)
    mid = len(embedding) // 2
    return embedding[:mid], embedding[mid:]
    
    

    
