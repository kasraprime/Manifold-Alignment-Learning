import numpy as np
import os
import sys
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import random
import subprocess

from models import MNISTEmbeddingNet, CIFAREmbeddingNet, RowNet
from datasets import uw_loaders, cifar10_loaders, mnist_loaders, gld_loaders
from utils import setup_dirs, setup_device, save_embeddings, load_embeddings, Visualize
from losses import cosine_pairwise_loss, deepcca

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_track', default=0, type=int)
    parser.add_argument('--experiment_name', default='RandomExperiment', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--task', default='uw', type=str)

    return parser.parse_known_args()

# Add learning rate scheduling. 
def lr_lambda(e):
    if e < 50:
        return 0.001
    elif e < 100:
        return 0.0001
    else:
        return 0.00001

def train(wandb_track, experiment_name, epochs, task, gpu_num=0, pretrained='', margin=0.4, losstype='deepcca'):
    """Train joint embedding networks."""
        
    epochs = int(epochs)
    gpu_num = int(gpu_num)
    margin = float(margin)
    
    # Setup the results and device.
    results_dir = setup_dirs(experiment_name)
    train_results_dir = os.path.join(results_dir, 'train_results/')
    if not os.path.exists(train_results_dir):
        os.makedirs(os.path.join(train_results_dir)
    device = setup_device(gpu_num)    

    #### Hyperparameters #####    
    #Initialize wandb
    if wandb_track==1:
        import wandb
        wandb.init(project=experiment_name)
        config = wandb.config
        config.epochs = epochs

    with open(os.path.join(results_dir, 'hyperparams_train.txt'), 'w') as f:
        f.write('Command used to run: python \n')
        f.write(f'ARGS: {ARGS}\n')
        f.write(f'device in use: {device}\n')
        f.write(f'--experiment_name {experiment_name}')
        f.write(f'--epochs {epochs}\n')
    
    # Setup data loaders and models.
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
            # Language
            model_A = RowNet(3072, embed_dim=1024)
            # Vision 
            model_B = RowNet(4096, embed_dim=1024)
    elif task == 'gld':
        train_loader, test_loader = gld_loaders('/home/iral/data_processing/gld_data_complete.pkl')
        model_A = RowNet(3072, embed_dim=1024)
        model_B = RowNet(4096, embed_dim=1024)
         
    # Finish model setup
    # If we want to load pretrained models to continue training...
    if pretrained == 'pretrained':        
        print('Starting from pretrained networks.')
        model_A.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_A_state.pt')))
        model_B.load_state_dict(torch.load(os.path.join(train_results_dir, 'model_B_state.pt')))
     
    print('Starting from scratch to train networks.')

    model_A.to(device)
    model_B.to(device)

    # Initialize the optimizers and loss function.
    optimizer_A = torch.optim.Adam(model_A.parameters(), lr=0.00001)
    optimizer_B = torch.optim.Adam(model_B.parameters(), lr=0.00001) 
    
    scheduler_A = torch.optim.lr_scheduler.LambdaLR(optimizer_A, lr_lambda)
    scheduler_B = torch.optim.lr_scheduler.LambdaLR(optimizer_B, lr_lambda)

    # Track batch losses.
    loss_hist = []
        
    # Put models into training mode.
    model_A.train()
    model_B.train()
    
    # Train.
    # wandb
    if wandb_track == 1:
        wandb.watch(model_A, log="all")
        wandb.watch(model_B, log="all")
    # for saving to files
    epoch_list = []
    loss_list = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        counter = 0
        for data in train_loader:
            data_a = data[0].to(device)
            data_b = data[1].to(device)
            #label = data[2]  

            # Zero the parameter gradients.
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            # Forward.                             
            if losstype == 'deepcca': # Based on Galen Andrew's Deep CCA
                # data_a is from domain A, and data_b is the paired data from domain B.                
                embedding_a = model_A(data_a)
                embedding_b = model_B(data_b)
                loss = deepcca(embedding_a, embedding_b, device, use_all_singular_values=True, outdim_size=128)
                                                               
            # Backward.                    
            loss.backward()
            
            # Update.
            optimizer_A.step()
            optimizer_B.step()

            # Save batch loss. Since we are minimizing -corr the loss is negative.
            loss_hist.append(-1 * loss.item())
            
            epoch_loss += embedding_a.shape[0] * loss.item()

            #reporting progress
            counter+=1
            if not counter % 64:
                print('epoch:',epoch, 'loss:', loss.item())
                if wandb_track == 1:
                    wandb.log({"epoch": epoch,"loss": loss})

        # Save network state at each epoch.
        torch.save(model_A.state_dict(), os.path.join(train_results_dir, 'model_A_state.pt'))
        torch.save(model_B.state_dict(), os.path.join(train_results_dir, 'model_B_state.pt'))
        
        #since the batch size is 1 therefore: len(trainloader)==counter
        print('*********** epoch is finished ***********')
        epoch_loss = -1 * epoch_loss
        print(f'epoch: {epoch}, loss(correlation): {epoch_loss / counter}')
        epoch_list.append(epoch + 1)
        loss_list.append(epoch_loss / counter)

        Visualize(
            os.path.join(train_results_dir, 'epoch_loss.pkl'),
            'Correlation History',
            True,
            'epoch',
            'Correlation (log scale)',
            None,
            'log',
            None,
            (14, 7),
            os.path.join(train_results_dir, 'Figures/')
        )
        # Update learning rate schedulers.
        scheduler_A.step()
        scheduler_B.step()

    with open(os.path.join(train_results_dir, 'epoch_loss.pkl'), 'wb') as fout:
        pickle.dump(([epoch_list, loss_list]), fout)

    # Plot and save batch loss history.
    with open(os.path.join(train_results_dir, 'epoch_corr.pkl'), 'wb') as fout:
        pickle.dump( ([loss_hist[::10]]), fout)

    Visualize(
        os.path.join(train_results_dir, 'epoch_corr.pkl'),
        'Correlation Batch',
        False,
        'Batch',
        'Correlation (log scale)',
        None,
        'log',
        None,
        (14, 7),
        os.path.join(train_results_dir, 'Figures/')
    )    

    #### Learn the transformations for CCA ####
    if losstype == "CCA":
        a_base = []
        b_base = []
        no_model = True

        if no_model: # without using model: using raw data without featurization
            for data in train_loader:    
                x = data[0].to(device)
                y = data[1].to(device)        
                if task=='uw':
                    a_base.append(x)
                    b_base.append(y)
                else:
                    a_base.append(x.cpu().detach().numpy())
                    b_base.append(y.cpu().detach().numpy())
        else:
            import torchvision.models as models
            #Either use these models, or use trained models with triplet loss
            res18_model = models.resnet18(pretrained=True)
            #changing the first layer of ResNet to accept images with 1 channgel instead of 3.
            res18_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
            # Select the desired layers
            model_A = torch.nn.Sequential(*list(res18_model.children())[:-2])
            model_B = torch.nn.Sequential(*list(res18_model.children())[:-2])
            model_A.eval()
            model_B.eval()
            for data in train_loader:                  
                x = data[0].to(device) # Domain A
                y = data[1].to(device) # Domain B            
                a_base.append(model_A(x).cpu().detach().numpy())
                b_base.append(model_B(y).cpu().detach().numpy())
        
        # Concatenate predictions.
        a_base = np.squeeze(np.concatenate(a_base, axis=0))
        b_base = np.squeeze(np.concatenate(b_base, axis=0))

        if no_model:
            new_a_base=[]
            new_b_base=[]
            for i in range(len(a_base)):
                new_a_base.append(a_base[i,:,:].flatten())
                new_b_base.append(b_base[i,:,:].flatten())
            new_a_base=np.asarray(new_a_base)
            new_b_base=np.asarray(new_b_base)
            a_base=new_a_base
            b_base=new_b_base
            
            print('Finished reshaping data, the shape is:', new_a_base.shape)                                        

        from sklearn.cross_decomposition import CCA
        from joblib import dump
        components=128
        cca = CCA(n_components=components)
        cca.max_iter = 5000
        cca.fit(a_base, b_base)
        dump(cca, 'Learned_CCA.joblib') 
    #### End of CCA fit to find the transformations ####

    print('Training Done!')

def main():
    ARGS, unused = parse_args()

    train(ARGS.wandb_track, ARGS.experiment_name, ARGS.epochs, args.ARGS)

if __name__ == '__main__':
    main() 
