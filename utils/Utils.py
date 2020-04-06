import math
import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from utils.PlotUtils import imshow

"""For CIFAR10 RL Net - Dog Bird Experiment"""


def create_datasets(trainloader, testloader, class1_ind, class2_ind):
    train_class1, train_class2 = [], []
    test_class1, test_class2 = [], []

    for i, data in enumerate(trainloader, 0):
        # get inputs ; data is a list of [inputs,labels]
        inputs, labels = data
        labels_np = labels.numpy()
        if labels_np[0] == class1_ind:
            train_class1.append(inputs)
        if labels_np[0] == class2_ind:
            train_class2.append(inputs)

    for i, data in enumerate(testloader, 0):
        # get inputs ; data is a list of [inputs,labels]
        inputs, labels = data
        labels_np = labels.numpy()
        if labels_np[0] == class1_ind:
            test_class1.append(inputs)
        if labels_np[0] == class2_ind:
            test_class2.append(inputs)

    train_class1 = torch.cat(train_class1)
    train_class2 = torch.cat(train_class2)
    test_class1 = torch.cat(test_class1)
    test_class2 = torch.cat(test_class2)

    print(str('Train: ' + str(train_class1.shape[0]) + ' Test: ' + str(test_class1.shape[0]) + ' added to class 1'))
    print(str('Train: ' + str(train_class2.shape[0]) + ' Test: ' + str(test_class2.shape[0]) + ' added to class 2'))

    return train_class1, train_class2, test_class1, test_class2


def loss_Cifar10(predicted_vals, rewards, taus=None, distributional=False):
    if distributional is False:
        delta = rewards - predicted_vals
        td_loss = torch.mean(0.5 * torch.pow(delta, 2))
        td_loss_ind_mean = None
    else:
        delta = rewards[:, None] - predicted_vals
        weights = torch.abs(taus[None] - torch.tensor(delta <= 0., dtype=torch.float32))
        td_loss_ind = weights * torch.abs(delta)
        td_loss_ind_mean = torch.mean(td_loss_ind, dim=0)
        td_loss = torch.mean(td_loss_ind_mean)

    return td_loss, td_loss_ind_mean


def create_taus(num_atoms):
    taus_np = np.linspace(0., 1., num_atoms + 2)[1:-1]
    taus = torch.tensor(taus_np, dtype=torch.float32)

    return taus


def sample_trial(class1, class2, pi1, pi2, rwd1, rwd2, prob1, trial_batch_size, do_plot=False):
    images = []
    rewards = []
    classes = []
    if do_plot is True:
        fig = plt.figure(figsize=(8, 8))
    num_channels, width, height = class1[0].shape
    for ib in range(trial_batch_size):
        if np.random.random() < prob1:
            images.append(class1[np.random.randint(len(class1))])
            rewards.append(rwd1[0] if np.random.random() < pi1 else rwd1[-1])
            classes.append(1)
            if do_plot is True and len(images) == 1:
                plt.subplot(1, 2, 1)
                imshow(class1[np.random.randint(len(class1))])
                plt.title('Class 1')
        else:
            images.append(class2[np.random.randint(len(class2))])
            rewards.append(rwd2[0] if np.random.random() < pi2 else rwd2[-1])
            classes.append(2)
            if do_plot is True and len(images) == 1:
                plt.subplot(1, 2, 2)
                imshow(class2[np.random.randint(len(class2))])
                plt.title('Class 2')

    images = torch.cat(images)
    images = images.view(trial_batch_size, num_channels, width, height)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    return images, rewards


"""For Saving """


def save_models(save_dir, save_name, net):  # Saving Net
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(net.state_dict(), os.path.join(save_dir, save_name))


def save_dict(save_dir, save_name, dict_save):  # Saving dictonary
    js = json.dumps(dict_save)
    f = open(os.path.join(save_dir, save_name), "w")
    f.write(js)
    f.close()


"""For DDRQN """


def conv2d_size_out(size, kernel_size=5, stride=2):
    # Number of Linear input connections depends on output of conv2d layers
    # and therefore the input image size, so compute it.

    return (size - (kernel_size - 1) - 1) // stride + 1


def compute_embedding_dim(w, h):
    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
    linear_input_size = convw * convh * 32  # embedding_dim

    return linear_input_size


def get_batch_size(states=None, state_embeddings=None):
    """ Batch size in base of input states"""
    batch_size = states.shape[0] if states is not None \
        else state_embeddings.shape[0]

    return batch_size


def get_transition():
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    return Transition


class LinearAnnealer:

    def __init__(self, start_value, end_value, num_steps):
        assert num_steps > 0 and isinstance(num_steps, int)

        self.steps = 0
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps

        self.a = (self.end_value - self.start_value) / self.num_steps
        self.b = self.start_value

    def step(self):
        self.steps = min(self.num_steps, self.steps + 1)

    def get(self):
        assert 0 < self.steps <= self.num_steps
        return self.a * self.steps + self.b


class LinearAnnealer2:
    def __init__(self, config):
        self.steps = 0
        self.EPS_END = config['EPS_END']
        self.EPS_START = config['EPS_START']
        self.EPS_DECAY = config['EPS_DECAY']
        self.num_steps = config['EPS_DECAY_STEPS']

    def step(self):
        self.steps = min(self.num_steps, self.steps + 1)

    def get(self):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps / self.EPS_DECAY)

        return eps_threshold
