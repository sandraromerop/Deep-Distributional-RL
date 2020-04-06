import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Utils import compute_embedding_dim, get_batch_size


# noinspection SpellCheckingInspection
"Module of  RL Networks  ,  BaseDQN taken from Mnih 2015 "
"Written by S.Romero 3/2020"


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SingleLayer(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size=3,  bias=False,
                 batch_norm=False):
        super(SingleLayer, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(n_channels_in,
                               n_channels_out,
                               kernel_size=kernel_size,
                               padding=1,
                               bias=bias)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(n_channels_out)

    def forward(self, x):
        if self.batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        return out


class DQNBase(nn.Module):

    def __init__(self, num_channels, h, w, outputs, batch_norm=False):

        super(DQNBase, self).__init__()

        if batch_norm:
            self.net = nn.Sequential(
                nn.Conv2d(num_channels, 16, kernel_size=5, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                Flatten())
        else:
            self.net = nn.Sequential(
                nn.Conv2d(num_channels, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=2),
                nn.ReLU(),
                Flatten())

    def forward(self, states):
        # Sandra 03/28: not to have  head--> this will go to DRDQN
        x = self.net(states)

        return x


class QRDQN(nn.Module):

    def __init__(self, num_channels, screen_height, screen_width, n_actions, num_quantiles=200,
                 batch_norm=False):
        super(QRDQN, self).__init__()

        self.num_quantiles = num_quantiles
        self.num_channels = num_channels
        self.num_actions = n_actions
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.embedding_dim = compute_embedding_dim(screen_height, screen_width)
        self.batch_norm = batch_norm
        # Extractor of DQN
        self.dqn_net = DQNBase(num_channels, screen_height, screen_width, n_actions,
                               batch_norm=batch_norm)

        # Quantile network
        self.q_net = nn.Sequential(nn.Linear(self.embedding_dim, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, n_actions * num_quantiles))

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = get_batch_size(states, state_embeddings)
        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)
        quantiles = self.q_net(state_embeddings).view(batch_size, self.num_quantiles, self.num_actions)

        assert quantiles.shape == (batch_size, self.num_quantiles, self.num_actions)

        return quantiles

    def calculate_q(self, states=None, state_embeddings=None):
        """Calculates expected value Q from quantiles"""
        assert states is not None or state_embeddings is not None
        batch_size = get_batch_size(states, state_embeddings)
        q = self(states=states, state_embeddings=state_embeddings).mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q


class CifarNet(nn.Module):
    """Basic Cifar net from Pytorch tutorial """

    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Cifar10ValueNet(nn.Module):

    def __init__(self,
                 num_atoms,
                 initializers=None,
                 regularizers=None,
                 partitioners=None,
                 name="cifar10_convnet"):

        super(Cifar10ValueNet, self).__init__()

        self.name = name
        self.num_atoms = num_atoms
        self.output_channels = [64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        self.num_layers = len(self.output_channels)
        self.kernel_shapes = [3] * self.num_layers  # All kernels are 3x3.
        self.strides = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1]
        self.initializers = initializers
        self.regularizers = regularizers
        self.partitioners = partitioners
        self.input_channels = [3]
        for i_out in self.output_channels:
            self.input_channels.append(i_out)
        self.input_channels = self.input_channels[0:len(self.output_channels)]
        self.net = self.make_net()

    def make_net(self):
        layers = []
        for i in range(self.num_layers):
            # In the bird dog experiment: seems like they only did layers of Conv2d,
            # no batchnorm and no pooling until the end that they take a 'reduction mean' for
            # flattening output
            layers.append(SingleLayer(self.input_channels[i],
                                      self.output_channels[i],
                                      kernel_size=self.kernel_shapes[i],
                                      bias=False, batch_norm=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        # Changed it to have aas dimensions the final dimension
        # the dimension of the final conv net (batch_sizx512)
        flat_output = torch.mean(x, dim=[2, 3])
        # From birddog experiment
        # Equivalent to :  tf.reduce_mean(net, reduction_indices=[1, 2], keepdims=False, name="avg_pool")
        linear = nn.Linear(flat_output.shape[-1], self.num_atoms)
        values = linear(flat_output)
        # Replace classifier output with linear function predicting values
        # torch.nn.Linear(in_features, out_features, bias=True)
        # Equivalent to: check https://sonnet.readthedocs.io/en/latest/api.html#linear-modules
        # values = snt.Linear(self._num_atoms,initializers=self._initializers,regularizers=self._regularizers,
        #   partitioners=self._partitioners)(flat_output)

        return flat_output, values

