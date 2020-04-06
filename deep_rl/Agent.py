import os
import random
import torch.optim as optim
from deep_rl.Memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
from utils.TensorboardUtils import RunningMeanStats
from utils.Utils import LinearAnnealer2
from utils.GymUtils import *
from deep_rl.Network import QRDQN
from collections import namedtuple

"Module of  RL agents - Written by S.Romero 3/2020 ,  BaseDQN taken from Mnih 2015"


class BaseAgent:

    def __init__(self, env, log_dir, config):

        self.env = env
        self.test_env = env
        seed = config['seed']
        cuda = config['cuda']

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(10000)
        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(config['log_interval'])

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_actions = self.env.action_space.n
        self.num_steps = 5 * (10 ** 7)
        self.batch_size = config['batch_size']
        self.log_interval = config['log_interval']
        self.eval_interval = config['eval_interval']
        self.num_eval_steps = config['num_eval_steps']
        self.gamma_n = config['GAMMA'] ** config['multi_step']
        self.start_steps = config['start_steps']
        self.epsilon_train = LinearAnnealer2(config)
        self.epsilon_eval = config['epsilon_eval']
        self.update_interval = config['update_interval']
        self.target_update_interval = config['target_update_interval']
        self.max_episode_steps = config['max_episode_steps']
        self.grad_cliping = config['grad_cliping_Base']
        self.policy_net = None
        self.target_net = None

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_greedy(self, eval_mode=False):
        if eval_mode:
            return np.random.rand() < self.epsilon_eval
        else:
            return self.steps < self.start_steps or np.random.rand() < self.epsilon_train.get()

    def explore(self):
        # Act with randomness.
        action = torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        return action

    def exploit(self, state):
        # Act without randomness.
        with torch.no_grad():
            action = self.policy_net.calculate_q(states=state).max(1)[1].view(1, 1)
        return action

    def train_episode(self):
        self.policy_net.train()
        self.target_net.train()
        # ----- Initialize counters
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0
        done = False
        # ----- Reset env and get state
        self.env.reset()
        current_screen = get_screen(self.env,
                                    self.device)  # last_screen = get_screen() #  state = current_screen - last_screen
        state = current_screen
        # ---- Start an episode (inner loop is happening within an episode until done)
        while (not done) and episode_steps <= self.max_episode_steps:

            if self.is_greedy(eval_mode=False):
                action = self.explore()
            else:
                action = self.exploit(state)
            # ----- Execute action
            _, reward, done, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device=self.device)

            # ----- Go to next state
            current_screen = get_screen(self.env, self.device)  # last_screen = current_screen
            if not done:
                next_state = current_screen  # - last_screen # next_state = current_screen
                done = torch.tensor([0], device=self.device)
            else:
                next_state = None
                done = torch.tensor([1], device=self.device)
            # ----- Store the transition in memory
            if next_state is not None:
                self.memory.push(state, action, next_state, reward, done)
            # ----- Counters
            self.steps += 1
            episode_steps += 1
            episode_return += reward
            # ----- Train
            state = next_state
            self.train_step_interval()

        # ----- We log running mean of stats.
        self.train_return.append(episode_return)

        # ----- We log evaluation results along with training frames = 4 * steps.
        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar('return/train', self.train_return.get(), 4 * self.steps)

        print('episodes:' + str(self.episodes) + '     episode steps:' + str(episode_steps) +
              '     return:' + str(episode_return.numpy()))

    def train_step_interval(self):
        self.epsilon_train.step()
        if self.steps % self.target_update_interval == 0:
            self.update_target_net()
        if self.steps % self.update_interval == 0 and self.steps >= self.start_steps:
            self.learn()
        if self.steps % self.eval_interval == 0:
            self.policy_net.eval()
            self.evaluate()
            self.save_models(os.path.join(self.model_dir, 'final'))
            self.policy_net.train()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self):
        raise NotImplementedError

    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                if self.is_greedy(eval_mode=True):
                    action = self.explore()
                else:
                    action = self.exploit(state)

                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        # We log evaluation results along with training frames = 4 * steps.
        self.writer.add_scalar(
            'return/test', mean_return, 4 * self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.policy_net.state_dict(),
            os.path.join(save_dir, 'online_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth'))


#  def __del__(self):
#     self.env.close()
#     self.test_env.close()
#     self.writer.close()


# noinspection SpellCheckingInspection
class QRDQNAgent(BaseAgent):

    def __init__(self, env, log_dir, config):
        super(QRDQNAgent, self).__init__(env, log_dir, config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = env.action_space.n
        self.num_channels = 3
        self.num_quantiles = config['QUANTILES']
        self.kappa = config['KAPPA']
        self.grad_cliping = config['grad_cliping_QRDQN']
        self.config = config
        # ----- Fixed fractions for quantiles
        taus = torch.arange(0, self.num_quantiles + 1, device=self.device, dtype=torch.float32) / self.num_quantiles
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, self.num_quantiles)
        init_screen = get_screen(env, self.device)
        _, _, screen_height, screen_width = init_screen.shape
        # ----- Create nets
        self.policy_net = QRDQN(self.num_channels, screen_height, screen_width,
                                self.n_actions, self.num_quantiles, batch_norm=False).to(self.device)
        self.target_net = QRDQN(self.num_channels, screen_height, screen_width,
                                self.n_actions, self.num_quantiles, batch_norm=False).to(self.device)
        # ----- Copy parameters of the policy_net to the target_net and Disable gradients for target_net
        self.update_target_net()
        for param in self.target_net.parameters():
            param.requires_grad = False
        # ----- Create Optimizer
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        # ----- Create Memory
        self.memory = ReplayMemory(10000)

    def learn(self):
        """Gets a batch of samples from memory, computes loss in the batch, updates params based on loss """
        self.learning_steps += 1
        # transition = get_transition()
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
        # ----- Get batch sample
        transitions = self.memory.sample(self.config['BATCH_SIZE'])
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        # non_final_mask.view(self.config['BATCH_SIZE'], 1, self.config['QUANTILES'])
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        # ----- Calculate quantile loss
        quantile_loss, mean_q = self.calculate_loss(state_batch, action_batch, reward_batch, non_final_next_states,
                                                    non_final_mask)
        # ----- Update parameters
        self.update_params(self.optimizer, quantile_loss, networks=[self.policy_net], retain_graph=False,
                           grad_cliping=self.grad_cliping)
        # ----- Write scalars on tensorboard
        if 4 * self.steps % self.log_interval == 0:
            self.writer.add_scalar('loss/quantile_loss', quantile_loss.detach().item(), 4 * self.steps)
            self.writer.add_scalar('stats/mean_Q', mean_q, 4 * self.steps)

    def calculate_loss(self, state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask):
        """Calculates loss based on a transition batch"""
        # ----- Calculate quantile values of current states and actions at taus.
        current_sa_quantiles = self.evaluate_sa_quantiles(self.policy_net, state_batch, action_batch)
        assert current_sa_quantiles.shape == (self.batch_size, self.num_quantiles, 1)
        # ----- Calculate quantile values of next states (discounted by gamma and added with reward)
        with torch.no_grad():
            target_sa_quantiles, next_q = self.evaluate_target_sa_quantiles(self.target_net, non_final_next_states,
                                                                            non_final_mask, reward_batch)
        # ----- Calculate td error
        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (self.batch_size, self.num_quantiles, self.num_quantiles)
        # ----- Calculate quantile huber loss based on td error
        quantile_huber_loss = self.calculate_quantile_huber_loss(td_errors, self.tau_hats, self.kappa)
        return quantile_huber_loss, next_q.detach().mean().item()

    def evaluate_sa_quantiles(self, network, states, actions):
        """Called : evaluate_quantile_at_action before-- gets quantiles at executed actions"""
        s_quantiles = network(states=states)
        assert s_quantiles.shape[0] == actions.shape[0]
        # Expand actions into (batch_size, N, 1).
        action_index = actions[..., None].expand(self.batch_size, self.num_quantiles, 1)
        # Calculate quantile values at specified actions.
        sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

        return sa_quantiles

    def evaluate_target_sa_quantiles(self, network, next_states, non_final_mask, rewards):
        """Called : gets quantiles at next states based on executed action and target_net"""
        next_q = network.calculate_q(states=next_states)
        # Calculate greedy actions.
        next_actions = torch.argmax(next_q, dim=1, keepdim=True)
        assert next_actions.shape == (self.batch_size, 1)
        # Calculate quantile values of next states and actions at tau_hats.
        next_sa_quantiles_temp = self.evaluate_sa_quantiles(self.target_net, next_states, next_actions)
        next_sa_quantiles_temp = next_sa_quantiles_temp.transpose(1, 2)
        next_sa_quantiles = torch.zeros(self.config['BATCH_SIZE'], 1, self.config['QUANTILES'])
        # non_final_mask = non_final_mask.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.config['QUANTILES'])
        next_sa_quantiles[non_final_mask] = next_sa_quantiles_temp
        assert next_sa_quantiles.shape == (self.config['BATCH_SIZE'], 1, self.config['QUANTILES'])
        # Calculate full target_sa_quantiles considering reward
        rewards = rewards.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.config['QUANTILES'])
        target_sa_quantiles = rewards + self.config['GAMMA'] * next_sa_quantiles
        assert target_sa_quantiles.shape == (self.config['BATCH_SIZE'], 1, self.config['QUANTILES'])

        return target_sa_quantiles, next_q

    def calculate_huber_loss(self, td_errors, kappa=1.0):
        """Get huber loss"""
        return torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa))

    def calculate_quantile_huber_loss(self, td_errors, taus, kappa=1.0):  # TO DO : qrdqn_agent: double check if correct
        """Get quantile huber loss"""
        assert not taus.requires_grad
        batch_size, N, N_dash = td_errors.shape

        # Calculate huber loss element-wisely.
        element_wise_huber_loss = self.calculate_huber_loss(td_errors, kappa)
        assert element_wise_huber_loss.shape == (batch_size, N, N_dash)

        # Calculate quantile huber loss element-wisely.
        element_wise_quantile_huber_loss = torch.abs(
            taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
        assert element_wise_quantile_huber_loss.shape == (
            batch_size, N, N_dash)

        return element_wise_quantile_huber_loss.sum(dim=1).mean()

    def update_params(self, optimizer, loss, networks, retain_graph=False, grad_cliping=None):
        """Update parameters: Optimizer steps"""
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        # Clip norms of gradients to stebilize training.
        # if grad_cliping is not None:
        #    for net in networks:
        #        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
        optimizer.step()



























































