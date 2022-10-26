
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch
from torch.linalg import multi_dot
import matplotlib.pyplot as plt
import datetime
import torch.nn.functional as F

import sys
sys.path.append("..")
from scripts.data_example_mask import get_sample_and_variables, get_input_tensor, get_posteior
from utils.model_mask import Net_mask
import scripts.data_example_mask as de


class Data(Dataset):
    def __init__(self, sample):
        self.data = torch.tensor(sample, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def custom_loss(output, input_, Q_m, sigma2_eps, l2_lambda, model):
    n_param = Q_m.shape[0]
    d = input_[:,:n_param]
    mask = input_[:,n_param:]
    mask = torch.flatten(mask.to(torch.long))
    ind = torch.nonzero(mask)
    ind = torch.flatten(ind)
    G = F.one_hot(ind, num_classes=n_param)
    G = torch.as_tensor(G, dtype=torch.float)
    Psi = output
    GPsi = multi_dot([output, torch.transpose(G, 0, 1)])
    Gd = multi_dot([d, torch.transpose(G, 0, 1)])

def custom_loss2(output, input_, Q_m, sigma2_eps, l2_lambda, model):
    Psi = output
    n_param = Q_m.shape[0]
    Gd = input_[:,:n_param]
    mask = input_[:,n_param:]
    GPsi = Psi*mask

    # Data loss
    squared_error = torch.square(GPsi - Gd)
    squared_error_sum = torch.sum(squared_error, [0, 1])
    likelihood_loss = squared_error_sum/sigma2_eps

    # Prior loss
    Q_m = torch.as_tensor(Q_m, dtype=torch.float)
    PsiQ_m = torch.matmul(Psi, Q_m)
    PsiQ_m = torch.unsqueeze(PsiQ_m, 1)
    Psi = torch.unsqueeze(Psi, 2)
    PsiQ_mPsi = torch.bmm(PsiQ_m, Psi).squeeze(2)
    prior_loss = torch.sum(PsiQ_mPsi)

    # Regularization loss
    l2_regularization_loss = torch.tensor(0, dtype=torch.float)
    loss_penalty = 0
    for p in model.parameters():
        l2_regularization_loss += torch.norm(p)
    loss_penalty += l2_lambda * l2_regularization_loss

    # Total loss per batch
    loss = likelihood_loss + prior_loss + loss_penalty
    return loss


if __name__ == "__main__":
    batch_size = 128
    n_batches = 10000
    N = batch_size*n_batches

    n_epochs = 2

    epoch_loss = np.zeros(n_epochs)
    l2_lambda = 0.1


    model = Net_mask()
    lr = 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    D = np.array(
        [[1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0,  1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0,  0, 1, -1, 0, 0, 0, 0, 0],
        [0, 0,  0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0,  0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0,  0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0,  0, 0, 0, 0, 0, 1, -1, 0],
        [0, 0,  0, 0, 0, 0, 0, 0, 1, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1],]
    )
    sigma2_eps = 0.001
    n_param = D.shape[0]
    Q_m = D.T@D
    mu_m = np.zeros(n_param)
    Q_m_modified = Q_m + np.identity(10)*0.001
    Sigma_m = np.linalg.inv(Q_m_modified)
    Sigma_eps = np.identity(n_param)*sigma2_eps
    Sigma_d = Sigma_m + Sigma_eps

    # loop over the dataset multiple times
    counter = 0
    losses = []

    for epoch in range(n_epochs):
        d_sample = np.random.multivariate_normal(mu_m.flatten(), Sigma_d, size=N)
        combined_sample = np.zeros((N, 2*n_param))
        mask = np.zeros((N, n_param))
        range_vector = np.zeros((N, n_param)) 
        range_vector[:,] = np.arange((n_param))
        n_masked = np.random.randint(n_param, size=N)
        n_masked = n_masked.reshape((N, 1))
        bool_vector = range_vector[:,] < n_masked
        mask[bool_vector] = 1
        np.random.shuffle(mask.T)
        d_sample = d_sample*mask
        combined_sample[:,:n_param] = d_sample
        combined_sample[:,n_param:] = mask

        data = torch.from_numpy(combined_sample)
        data = data.type(torch.float)
        epoch_running_loss = 0
        running_loss = 0.0
        i_prev = 0
        for i in range(1, N, 128):
            input_ = data[i_prev:i,:]
            i_prev = i

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_)

            loss = custom_loss2(outputs, input_, Q_m, sigma2_eps, l2_lambda=l2_lambda, model=model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            freq = 1000
            if counter % freq == 0:
                print(f'[{epoch + 1}, {counter + 1:5d}] loss: {running_loss/freq/batch_size:20.10f}')
                losses.append(running_loss)
                running_loss = 0.0
            counter += 1
            epoch_running_loss += running_loss

        epoch_loss[epoch] = epoch_running_loss
        

    print('Finished Training')

    plt.plot(range(len(losses)), losses)
    plt.yscale('log',base=10)
    plt.savefig('../saved_models_mask3/loss.pdf')
    plt.show()

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name = "../saved_models_mask/model_weights_mask_" + time + ".pth"
    torch.save(model.state_dict(), name)

    # Constructing example to plot when finished
    d = [1, -5, 3, 2, -1]
    x = [2, 3, 4, 6, 9]

    input_tensor = get_input_tensor(d, x)
    out = model(input_tensor)

    with torch.no_grad():
        out = out.numpy().flatten()

    post_mean, _ = get_posteior(d, x)
    plt.plot(range(10), post_mean)
    plt.plot(range(10), out)
    plt.legend(["Analytical solution", "Neural net estimate"])
    plt.plot(x, [out[i] for i in x], 'o')
    plt.plot(x, [post_mean[i] for i in x], 'o')
    plt.title("Analystical solution vs neural net estimate")
    plt.xlabel("m")
    plt.ylabel("m|d")
    plt.savefig('../saved_models_mask3/plot3.pdf')
    plt.show()