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
import scripts.data_example as de


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
    batch_size = 1
    n_batches = 1000000
    N = batch_size*n_batches

    #data_indices, mu_m, Sigma_eps, G, mu_eps, Sigma_m, Q_m, Sigma_d, sigma2_eps = de.get_example_variables()
    #d_sample = np.random.multivariate_normal(G@mu_m.T.flatten(), Sigma_d, size=N)

    combined_sample, Sigma_m, Q_m, sigma2_eps = get_sample_and_variables(N)

    n_epochs = 10
    epoch_loss = np.zeros(n_epochs)
    l2_lambda = 0.05

    dataset = Data(combined_sample)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Net_mask()
    lr = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # loop over the dataset multiple times
    for epoch in range(n_epochs):

        epoch_running_loss = 0
        running_loss = 0.0
        for i, input_ in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = dat

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_)

            loss = custom_loss(outputs, input_, Q_m, sigma2_eps, l2_lambda=l2_lambda, model=model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            epoch_running_loss += running_loss
            running_loss = 0.0
        epoch_loss[epoch] = epoch_running_loss

        

    print('Finished Training')

    print(epoch_loss)
    plt.plot(range(n_epochs), epoch_loss)
    plt.show()

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    #name = "../saved_models_mask/model_weights_" + time + ".pth"
    name = "../saved_models_mask/model_weights_mask.pth"
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
    plt.show()