import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_quantiles(current_sa_quantiles, tau_hats, config):
    plt.figure()
    plt.clf()
    plt.title('Quantiles')
    plt.xlabel('Quantile')
    plt.ylabel('Val')
    th = tau_hats.clone().detach().requires_grad_(False)
    for i in range(config.BATCH_SIZE):
        # saq = torch.tensor(current_sa_quantiles[i,:,:], dtype=torch.float)
        saq = current_sa_quantiles[i, :, :].clone().detach().requires_grad_(False)
        plt.plot(tau_hats.view(200, -1), saq.view(200, -1))


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

