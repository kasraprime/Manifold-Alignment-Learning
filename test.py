import os
import sys
import numpy as np
import argparse
import csv 
from numpy.linalg import norm
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
import random
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import roc_curve, classification_report, precision_recall_fscore_support
from metrics import corr_between, rank_from, rank_from_class, knn, mean_reciprocal_rank, object_identification_task_rank, object_identification_task_classifier
import torch
import seaborn as sns
sns.set_context('poster')
import matplotlib.pyplot as plt
#%matplotlib inline

from models import MNISTEmbeddingNet, CIFAREmbeddingNet, RowNet
from datasets import cifar10_loaders, mnist_loaders, uw_loaders
from utils import setup_dirs, setup_device, save_embeddings, load_embeddings, Visualize

def test(experiment_name, task, gpu_num=0, pretrained='', margin=0.4, losstype='deepcca'):
    cosined=False    
    embed_dim=1024
    gpu_num = int(gpu_num)
    margin = float(margin)
    
    # Setup the results and device.
    results_dir = setup_dirs(experiment_name)
    if not os.path.exists(results_dir+'test_results/'):
        os.makedirs(results_dir+'test_results/')
    test_results_dir=results_dir+'test_results/'

    device = setup_device(gpu_num)    

    #### Hyperparameters #####
    #Initialize wandb
    #import wandb
    #wandb.init(project=experiment_name)
    #config = wandb.config


    with open(results_dir+'hyperparams_test.txt','w') as f:        
        f.write('Command used to run: python ')
        f.write(' '.join(sys.argv))
        f.write('\n')
        f.write('device in use: '+str(device))
        f.write('\n')
        f.write('--experiment_name '+str(experiment_name))
        f.write('\n')
        

    # Setup data loaders and models based on task.
    if task == 'cifar10':
        train_loader, test_loader = cifar10_loaders()
        model_A = CIFAREmbeddingNet()
        model_B = CIFAREmbeddingNet()
    elif task == 'mnist':
        train_loader, test_loader = mnist_loaders()
        model_A = MNISTEmbeddingNet()
        model_B = MNISTEmbeddingNet()        
    elif task == 'uw':
        uw_data = 'bert'
        train_loader, test_loader = uw_loaders(uw_data)
        if uw_data == 'bert':
            model_A = RowNet(3072, embed_dim=1024) # Language.
            model_B = RowNet(4096, embed_dim=1024) # Vision.        

    # Finish model setup.
    model_A.load_state_dict(torch.load(results_dir+'train_results/model_A_state.pt'))
    model_B.load_state_dict(torch.load(results_dir+'train_results/model_B_state.pt'))
    model_A.to(device)
    model_B.to(device)
    # Put models into evaluation mode.
    model_A.eval()
    model_B.eval()

    """For UW data."""
    ## we use train data to calculate the threshhold for distance.
    a_train = []
    b_train = []
    # loading saved embeddings to be faster
    a_train=load_embeddings(test_results_dir+'lang_embeds_train.npy')
    b_train=load_embeddings(test_results_dir+'img_embeds_train.npy')

    # Iterate through the train data.
    if a_train is None or b_train is None:
        for data in train_loader:  
            anchor_data, positive_data, label = data      
            a_train.append(model_A(anchor_data.to(device)).cpu().detach().numpy())         
            b_train.append(model_B(positive_data.to(device)).cpu().detach().numpy()) 
    
    #saving embeddings if not already saved
    save_embeddings(test_results_dir+'lang_embeds_train.npy', a_train)        
    save_embeddings(test_results_dir+'img_embeds_train.npy', b_train)

    a_train = np.concatenate(a_train,axis=0)
    b_train = np.concatenate(b_train,axis=0)


    # Test data
    # For accumulating predictions to check embedding visually using test set.
    # a is embeddings from domain A, b is embeddings from domain B, ys is their labels
    a = []
    b = []
    ys = []
    instance_data=[]    

    # loading saved embeddings to be faster
    a = load_embeddings(test_results_dir+'lang_embeds.npy')
    b = load_embeddings(test_results_dir+'img_embeds.npy')    
    if a is None or b is None:        
        compute_test_embeddings= True
    
    # Iterate through the test data.
    for data in test_loader:          
        language_data, vision_data, object_name, instance_name = data
        language_data = language_data.to(device) 
        vision_data = vision_data.to(device)                     
        instance_data.extend(instance_name)
        if compute_test_embeddings:
            a.append(model_A(language_data).cpu().detach().numpy()) # Language.        
            b.append(model_B(vision_data).cpu().detach().numpy()) # Vision.        
        ys.extend(object_name)
        
    # Convert string labels to ints.
    labelencoder = LabelEncoder()
    labelencoder.fit(ys)
    ys = labelencoder.transform(ys)
        

    #saving embeddings if not already saved
    save_embeddings(test_results_dir+'lang_embeds.npy', a)
    save_embeddings(test_results_dir+'img_embeds.npy', b)        

    # Concatenate predictions.
    a = np.concatenate(a,axis=0)
    b = np.concatenate(b,axis=0)
    ab = np.concatenate((a,b),axis=0)
    

    ground_truth, predicted, distance = object_identification_task_classifier(a,b,ys,a_train,b_train,lamb_std=1,cosine=cosined)

    #### Retrieval task by giving an image and finding the closest word descriptions ####
    ground_truth_word, predicted_word, distance_word = object_identification_task_classifier(b,a,ys,b_train,a_train,lamb_std=1,cosine=cosined)
    with open('retrieval_non_pro.csv', mode='w') as retrieval_non_pro:
        csv_file_writer = csv.writer(retrieval_non_pro, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_file_writer.writerow(['image', 'language', 'predicted', 'ground truth' ])
        for i in range(50):        
            csv_file_writer.writerow([instance_data[0], instance_data[i], predicted_word[0][i], ground_truth_word[0][i]])        
        
    precisions = []
    recalls = []
    f1s = []
    precisions_pos = []
    recalls_pos = []
    f1s_pos = []
    #print(classification_report(oit_res[i], 1/np.arange(1,len(oit_res[i])+1) > 0.01))
    for i in range(len(ground_truth)):
        p,r,f,s = precision_recall_fscore_support(ground_truth[i], predicted[i],warn_for=(),average='micro') 
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        p,r,f,s = precision_recall_fscore_support(ground_truth[i], predicted[i],warn_for=(),average='binary') 
        precisions_pos.append(p)
        recalls_pos.append(r)
        f1s_pos.append(f)
    
    print('\n ')
    print(experiment_name+'_'+str(embed_dim))
    print('MRR,    KNN,    Corr,   Mean F1,    Mean F1 (pos only)')
    print('%.3g & %.3g & %.3g & %.3g & %.3g' % (mean_reciprocal_rank(a,b,ys,cosine=cosined), knn(a,b,ys,k=5,cosine=cosined), corr_between(a,b,cosine=cosined), np.mean(f1s), np.mean(f1s_pos)))

    plt.figure(figsize=(14,7))
    for i in range(len(ground_truth)):
        fpr, tpr, thres = roc_curve(ground_truth[i], [1-e for e in distance[i]], drop_intermediate=True)
        plt.plot(fpr,tpr,alpha=0.08,color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')    
    plt.savefig(test_results_dir+'_'+str(embed_dim)+'_ROC.svg')

    # Pick a pair, plot distance in A vs distance in B. Should be correlated.
    a_dists = []
    b_dists = []
    for _ in range(3000):
        i1 = random.randrange(len(a))
        i2 = random.randrange(len(a))
        a_dists.append(euclidean(a[i1], a[i2]))
        b_dists.append(euclidean(b[i1], b[i2])) 
    #     a_dists.append(cosine(a[i1], a[i2]))
    #     b_dists.append(cosine(b[i1], b[i2])) 
        
        
    # Plot.
    plt.figure(figsize=(14,14))
    #plt.title('Check Distance Correlation Between Domains')
    plt.xlim([0,3])
    plt.ylim([0,3])
    # plt.xlim([0,max(a_dists)])
    # plt.ylim([0,max(b_dists)])
    # plt.xlabel('Distance in Domain A')
    # plt.ylabel('Distance in Domain B')
    plt.xlabel('Distance in Language Domain')
    plt.ylabel('Distance in Vision Domain')
    #plt.plot(a_dists_norm[0],b_dists_norm[0],'.')
    #plt.plot(np.arange(0,2)/20,np.arange(0,2)/20,'k-',lw=3)
    plt.plot(a_dists,b_dists,'o',alpha=0.5)
    plt.plot(np.arange(0,600),np.arange(0,600),'k--',lw=3,alpha=0.5)
    #plt.text(-0.001, -0.01, 'Corr: %.3f'%(pearsonr(a_dists,b_dists)[0]),  fontsize=20)
    plt.savefig(test_results_dir+'_'+str(embed_dim)+'_CORR.svg')



    # Inspect embedding distances.
    clas = 5  # Base class.
    i_clas = [i for i in range(len(ys)) if ys[i].item() == clas]
    i_clas_2 = np.random.choice(i_clas, len(i_clas), replace=False)

    clas_ref = 4  # Comparison class.
    i_clas_ref = [i for i in range(len(ys)) if ys[i].item() == clas_ref]

    ac = np.array([a[i] for i in i_clas])
    bc = np.array([b[i] for i in i_clas])

    ac2 = np.array([a[i] for i in i_clas_2])
    bc2 = np.array([b[i] for i in i_clas_2])

    ac_ref = np.array([a[i] for i in i_clas_ref])
    aa_diff_ref = norm(ac[:min(len(ac),len(ac_ref))]-ac_ref[:min(len(ac),len(ac_ref))], ord=2, axis=1)

    ab_diff = norm(ac-bc2, ord=2, axis=1)
    aa_diff = norm(ac-ac2, ord=2, axis=1)
    bb_diff = norm(bc-bc2, ord=2, axis=1)

    # aa_diff_ref = [cosine(ac[:min(len(ac),len(ac_ref))][i],ac_ref[:min(len(ac),len(ac_ref))][i]) for i in range(len(ac[:min(len(ac),len(ac_ref))]))]

    # ab_diff = [cosine(ac[i],bc2[i]) for i in range(len(ac))]
    # aa_diff = [cosine(ac[i],ac2[i]) for i in range(len(ac))]
    # bb_diff = [cosine(bc[i],bc2[i]) for i in range(len(ac))]


    bins = np.linspace(0, 0.1, 100)

    plt.figure(figsize=(14,7))
    plt.hist(ab_diff,bins,alpha=0.5,label='between embeddings')
    plt.hist(aa_diff,bins,alpha=0.5,label='within embedding A')
    plt.hist(bb_diff,bins,alpha=0.5,label='within embedding B')

    plt.hist(aa_diff_ref,bins,alpha=0.5,label='embedding A, from class '+str(clas_ref))

    plt.title('Embedding Distances - Class: '+str(clas))
    plt.xlabel('L2 Distance')
    plt.ylabel('Count')
    plt.legend()


    #labelencoder.classes_
    classes_to_keep = [36, 6, 9, 46, 15, 47, 50, 22, 26, 28]
    print(labelencoder.inverse_transform(classes_to_keep))

    ab_norm = [e for i,e in enumerate(ab) if ys[i%len(ys)]  in classes_to_keep]
    ys_norm = [e for e in ys if e in classes_to_keep]

    color_index = {list(set(ys_norm))[i]: i for i in range(len(set(ys_norm)))} #set(ys_norm)
    markers = ["o","v","^","s","*","+","x","D","h","4"]
    marker_index = {list(set(ys_norm))[i]: markers[i] for i in range(len(set(ys_norm)))}

    
    embedding = umap.UMAP(n_components=2).fit_transform(ab_norm) # metric='cosine'
    # Plot UMAP embedding of embeddings for all classes.
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

    mid = len(ys_norm)

    ax1.set_title('Language UMAP')
    for e in list(set(ys_norm)):
        x1 = [embedding[:mid, 0][i] for i in range(len(ys_norm)) if ys_norm[i] == e]
        x2 = [embedding[:mid, 1][i] for i in range(len(ys_norm)) if ys_norm[i] == e]
        ax1.scatter(x1, x2, marker=marker_index[int(e)], alpha=0.5, c=[sns.color_palette("colorblind", 10)[color_index[int(e)]]],label=labelencoder.inverse_transform([int(e)])[0])
    ax1.set_xlim([min(embedding[:,0])-4, max(embedding[:,0])+4])
    ax1.set_ylim([min(embedding[:,1])-4, max(embedding[:,1])+4])
    ax1.grid(True)
    ax1.legend(loc='upper center', bbox_to_anchor=(1.1, -0.08),fancybox=True, shadow=True, ncol=5)

    ax2.set_title('Vision UMAP')
    for e in list(set(ys_norm)):
        x1 = [embedding[mid::, 0][i] for i in range(len(ys_norm)) if ys_norm[i] == e]
        x2 = [embedding[mid::, 1][i] for i in range(len(ys_norm)) if ys_norm[i] == e]
        ax2.scatter(x1, x2, marker=marker_index[int(e)], alpha=0.5, c=[sns.color_palette("colorblind", 10)[color_index[int(e)]]])
    ax2.set_xlim([min(embedding[:,0])-4, max(embedding[:,0])+4])
    ax2.set_ylim([min(embedding[:,1])-4, max(embedding[:,1])+4])
    ax2.grid(True)

    plt.savefig(test_results_dir+'_'+str(embed_dim)+'_UMAP_wl.svg', bbox_inches='tight')

    #sns.palplot(sns.color_palette("colorblind", 10))

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='RandomExperiment', type=str)    
    parser.add_argument('--task', default='uw', type=str)    
    args = parser.parse_args()
    test(args.experiment_name,args.task)