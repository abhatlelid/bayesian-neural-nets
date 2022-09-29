from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch
from torch.linalg import multi_dot
import matplotlib.pyplot as plt
import datetime

import sys
sys.path.append("..")
from utils.model import Net
import scripts.data_example as de


class Data(Dataset):
    def __init__(self, d_sample):
        self.data = torch.tensor(d_sample, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]


def custom_loss(output, target, Q_m, sigma2_eps, l2_lambda, model, G):
    Psi = output
    G = torch.as_tensor(G, dtype=torch.float)
    GPsi = multi_dot([output, torch.transpose(G, 0, 1)])

    # Data loss
    squared_error = torch.square(GPsi - target)
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
    batch_size = 64
    n_batches = 100
    N = batch_size*n_batches

    data_indices, mu_m, Sigma_eps, G, mu_eps, Sigma_m, Q_m, Sigma_d, sigma2_eps = de.get_example_variables()
    d_sample = np.random.multivariate_normal(G@mu_m.T.flatten(), Sigma_d, size=N)

    n_epochs = 10
    l2_lambda = 0.1

    dataset = Data(d_sample)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Net()
    lr = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # loop over the dataset multiple times
    for epoch in range(n_epochs):

        running_loss = 0.0
        for i, dat in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = dat

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = custom_loss(outputs, labels, Q_m, sigma2_eps, l2_lambda=l2_lambda, model=model, G=G)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    print('Finished Training')
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name = "../saved_models/model_weights_" + time + ".pth"
    torch.save(model.state_dict(), name)

    # Constructing example to plot when finished
    d = np.array([8, -7, 0.01])
    d = torch.tensor(d, dtype=torch.float)
    out = model(d)
    with torch.no_grad():
        out = out.numpy().flatten()

    plt.plot(range(0, 10), out)
    plt.plot(data_indices, [out[i] for i in data_indices], 'bo')
    plt.show()