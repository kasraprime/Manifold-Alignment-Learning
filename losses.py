import torch

def cosine_pairwise_loss(d_a,d_b,match,model_A,model_B):
    criterion = torch.nn.CosineEmbeddingLoss()
    l=criterion(model_A(d_a),model_B(d_b),torch.tensor(match))
    return l

def deepcca(embedding_a, embedding_b, device, use_all_singular_values, outdim_size):
    # The code is based on https://github.com/VahidooX/DeepCCA/blob/master/objectives.py    
    H1=embedding_a
    H2=embedding_b
    r1 = 1e-4
    r2 = 1e-4
    eps = 1e-12
    o1 = H1.size(0)
    o2 = H2.size(0)
    m = H1.size(1)

    H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
    H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
    
    SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
    SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,H1bar.t()) + r1 * torch.eye(o1, device=device)
    SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,H2bar.t()) + r2 * torch.eye(o2, device=device)
    
    # Calculating the root inverse of covariance matrices by using eigen decomposition
    [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
    [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

    # Added to increase stability
    posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]
    posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]
    
    SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
    SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())
    Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,SigmaHat12), SigmaHat22RootInv)

    if use_all_singular_values:
        # all singular values are used to calculate the correlation
        tmp = torch.trace(torch.matmul(Tval.t(), Tval))        
        corr = torch.sqrt(tmp)        
    else:
        # just the top self.outdim_size singular values are used
        U, V = torch.symeig(torch.matmul(Tval.t(), Tval), eigenvectors=True)
        # U = U[torch.gt(U, eps).nonzero()[:, 0]]
        U = U.topk(outdim_size)[0]
        corr = torch.sum(torch.sqrt(U))
    l= -corr
    return l
