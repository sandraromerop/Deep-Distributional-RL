class SingleLayer(nn.Module):
    def __init__(self, nChannelsIn, nChannelsOut, kernel_size=3, padding=1, bias=False,
                 batchNorm=False):
        super(SingleLayer, self).__init__()
        self.batchNorm =batchNorm
        self.conv1 = nn.Conv2d(nChannelsIn, nChannelsOut, kernel_size=3,
                               padding=1, bias=False)
        if batchNorm:
            self.bn1 = nn.BatchNorm2d(nChannelsOut)

    def forward(self, x):
        if self.batchNorm:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

net = Net()

loss_Cifar10
create_taus
sample_trial


#_-----------------------
with self._enter_variable_scope():
    self._layers = tuple(
        snt.Conv2D(
            name="conv_net_2d/conv_2d_{}".format(i),
            output_channels=self._output_channels[i],
            kernel_shape=self._kernel_shapes[i],
            stride=self._strides[i],
            padding=self._paddings[i],
            initializers=initializers,
            regularizers=regularizers,
            partitioners=partitioners,
            use_bias=True) for i in range(self._num_layers))
net = inputs
    for i, layer in enumerate(self._layers):
      net = layer(net)
      net = tf.nn.relu(net)

    flat_output = tf.reduce_mean(
        net, reduction_indices=[1, 2], keepdims=False, name="avg_pool")

    # Replace classifier output with linear function predicting values
    values = snt.Linear(
        self._num_atoms,
        initializers=self._initializers,
        regularizers=self._regularizers,
        partitioners=self._partitioners)(flat_output)

    return _Outputs(activations=flat_output, values=values)
#_-----------------------
def bird_dog_experiment(num_updates, num_trials):
    losses = np.zeros((2, num_trials, num_updates,))
    transfer_losses = np.zeros((2, num_trials, num_updates,))
    mserrors = np.zeros((2, num_trials, num_updates, 2))
    for trial in range(num_trials):




def trainCifar10(num_atoms, distributional, trial_batch_size, num_epochs,
                 class1, class2, pi1, pi2, rwd1, rwd2, prob1):
    time = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = os.path.join('../logs', 'CIFAR_10_TEST', f'{time}')
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    writer = SummaryWriter(log_dir=summary_dir)
    log_interval = 100
    value_net = Cifar10ValueNet(num_atoms)
    optimizer = optim.Adam(value_net.parameters(), lr=0.0005)
    taus = create_taus(num_atoms)
    running_loss = 0.0
    running_td_loss_cum = []
    running_mse_loss_cum = []
    steps = 0
    for epoc_nb in range(num_epochs):
        images, rewards = sample_trial(class1, class2, pi1, pi2, rwd1, rwd2, prob1, trial_batch_size)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        tdfeatures, output_values = value_net(images)
        td_loss = lossCifar10(output_values, rewards, taus=taus, distributional=distributional)
        td_loss.backward()
        optimizer.step()
        # print statistics
        running_loss += td_loss.item()
        running_td_loss_cum.append(td_loss)
        running_mse_loss_cum.append(mse)
        print('epoch num:' + str(epoc_nb))
        steps += 1
        if epoc_nb % log_interval == 0:
            writer.add_scalar('td_loss/train', td_loss.item(), 4 * steps)
        if epoc_nb % log_interval == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoc_nb + 1, epoc_nb + 1, running_loss / 2000))

            running_loss = 0.0
    print('Finished training')
    return tdfeatures, output_values, running_loss, running_td_loss_cum, running_mse_loss_cum


