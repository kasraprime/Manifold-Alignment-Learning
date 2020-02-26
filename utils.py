import os
import torch
import numpy as np
import scipy as sp
import pickle
import matplotlib.pyplot as plt

def setup_dirs(experiment_name):
    """Create results and experiment directories if doesn't exist."""
    results_dir = './results/' + experiment_name + '/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir
    
    
def setup_device(gpu_num=0):
    """Setup device."""
    device_name = 'cuda:'+str(gpu_num) if torch.cuda.is_available() else 'cpu'  # Is there a GPU? 
    device = torch.device(device_name)
    return device

def load_embeddings(embedding_path):
    embedding = None
    if os.path.exists(embedding_path):
        embedding = np.load(embedding_path)            
    return embedding

def save_embeddings(embedding_path,embedding):
    if not os.path.exists(embedding_path):
        np.save(embedding_path, embedding)

def Visualize(pkldata,title,has_x_axis,xlablel,ylabel,xscale,yscale,legend,figsize,location):
    if not os.path.exists(location):        
        os.makedirs(location)
    visual=pickle.load(open(pkldata, 'rb'))
    #Note: I always save the x label in visual[0], and the results from different methods in next indices visual[1], visual[2], ...
    if figsize is not None:
        plt.figure(figsize=figsize)
    if legend is not None:
        if has_x_axis:
            for i in range(len(visual)-1):         
                plt.plot(visual[0],visual[i+1],label=legend[i]) 
        else:
            for i in range(len(visual)):         
                plt.plot(visual[i],label=legend[i]) 

    else:
        if has_x_axis:            
            for i in range(len(visual)-1):         
                plt.plot(visual[0],visual[i+1]) 
        else:            
            for i in range(len(visual)):         
                plt.plot(visual[i]) 
    plt.title(title)
    plt.xlabel(xlablel)
    plt.ylabel(ylabel)
    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)
    if legend is not None:
        plt.legend(loc='lower right')
    plt.savefig(location+title+'.pdf')
    plt.close('all')

def procrustes_distance(A, B):
        
        # Translation.
        mean_A = torch.mean(A,0)
        mean_B = torch.mean(B,0)
        A = A - mean_A
        B = B - mean_B
        
        # Scaling.
        s_A = torch.norm(A)
        s_B = torch.norm(B)
        A = torch.div(A, s_A)
        B = torch.div(B, s_B)
        
        # Orthogonal Procrustes.
        M = torch.t(torch.mm(torch.t(B),A))
        u, s, v = torch.svd(M)
        R = torch.mm(u,torch.t(v))
        s = torch.sum(s)
        B = torch.mm(B,torch.t(R)) * s
        
        # Compute distance.
        dists = torch.norm(A-B, dim=1)
        
        return dists
    

class Procrustes():
    """Transformation: translate, scale, and rotate/reflect."""
    
    def __init__(self, A, B, trans=True, scale=True, rot=True):
        self.trans = trans
        self.scale = scale
        self.rot = rot
        
        if self.trans:
            # Compute translation.
            self.mean_A = np.mean(A, axis=0)
            self.mean_B = np.mean(B, axis=0)
            A = A - self.mean_A
            B = B - self.mean_B
        
        if self.scale:
            # Compute scaling.
            self.s_A = np.linalg.norm(A)
            self.s_B = np.linalg.norm(B)
            A = np.divide(A, self.s_A)
            B = np.divide(B, self.s_B)
        
        if self.rot:
            # Compute Orthogonal Procrustes.
            M = B.T.dot(A).T
            u, s, vh = np.linalg.svd(M)
            self.R = u.dot(vh)
            self.s = np.sum(s)
     
    def transform(self, A, B):
        if self.trans:
            A = A - self.mean_A
            B = B - self.mean_B
        if self.scale:
            A = np.divide(A, self.s_A)
            B = np.divide(B, self.s_B)
        if self.rot:
            B = B.dot(self.R.T) * self.s
        return A, B
