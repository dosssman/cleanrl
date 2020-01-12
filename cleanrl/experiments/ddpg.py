import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
import numpy as np
import gym
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

# MODIFIED: Import buffer with random batch sampling support
from cleanrl.buffers import SimpleReplayBuffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).strip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="HopperBulletEnv-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=1000,
                        help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=int(1e6),
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                        help='whether to set `torch.backends.cudnn.deterministic=True`')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether to use CUDA whenever possible')
    parser.add_argument('--prod-mode', type=bool, default=False,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--notb', action='store_true',
                        help='No Tensorboard logging')

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=int(1e5),
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-update-interval', type=int, default=1,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--batch-size', type=int, default=128,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--start-steps', type=int, default=int(1e4),
                        help='initial random exploration step count')

    # TODO: Add Parameter Noising support ~ https://arxiv.org/abs/1706.01905
    parser.add_argument('--noise-type', type=str, choices=["ou", "naive"], default="ou",
                        help='type of noise to be used when sampling exploratiry actions')
    parser.add_argument('--noise-std', type=float, default=0.1,
                        help="standard deviation of the Normal dist for action noise sampling")

    # Neural Network Parametrization
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[256,128,80])

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
input_shape, preprocess_obs_fn = preprocess_obs_space(env.observation_space, device)
output_shape = preprocess_ac_space(env.action_space)
act_limit = env.action_space.high[0]
assert isinstance(env.action_space, Box), "only continuous action space is supported"

# ALGO LOGIC: initialize agent here:
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# MODIFIED: Added noise function for exploration
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
# TODO: More elegant way ?
if args.noise_type == "ou":
    class OrnsteinUhlenbeckActionNoise():
        def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
            self.theta = theta
            self.mu = mu
            self.sigma = sigma
            self.dt = dt
            self.x0 = x0
            self.reset()

        def __call__(self):
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
            self.x_prev = x
            return x

        def reset(self):
            self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

        def __repr__(self):
            return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
    ou_act_noise = OrnsteinUhlenbeckActionNoise( mu=np.zeros(output_shape), sigma=float(args.noise_std) * np.ones(output_shape))

# Determinsitic policy
class Policy(nn.Module):
    def __init__(self, squashing = True):
        # Custom
        super().__init__()
        self.layers = nn.ModuleList()
        self.squashing = squashing

        current_dim = input_shape
        for hsize in args.hidden_sizes:
            self.layers.append(nn.Linear(current_dim, hsize))
            current_dim = hsize

        # TODO: Add squashing like mechanism for action bound inforcing
        self.fc_mu = nn.Linear(args.hidden_sizes[-1], output_shape)
        self.tanh_mu = nn.Tanh()

    def forward(self, x):
        # # Workaround Seg. Fault when passing tensor to Q Function
        if not isinstance(x, torch.Tensor):
            x = preprocess_obs_fn(x)

        for layer in self.layers:
            x = F.relu(layer(x))

        mus = self.fc_mu(x)

        return self.tanh_mu( mus)

class QValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

        current_dim = input_shape + output_shape
        for hsize in args.hidden_sizes:
            self.layers.append(nn.Linear(current_dim, hsize))
            current_dim = hsize

        self.layers.append(nn.Linear(args.hidden_sizes[-1], 1))

    def forward(self, x, a):
        # Workaround Seg. Fault when passing tensor to Q Function
        if not isinstance(x, torch.Tensor):
            x = preprocess_obs_fn(x)

        if not isinstance(a, torch.Tensor):
            a = preprocess_obs_fn(a)

        x = torch.cat([x,a], 1)

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        return self.layers[-1](x)

buffer = SimpleReplayBuffer(env.observation_space, env.action_space, args.buffer_size, args.batch_size)
buffer.set_seed(args.seed)

# Defining agent's policy and corresponding target
pg = Policy().to(device)
pg_target = Policy().to(device)

qf = QValue().to(device)
qf_target = QValue().to(device)

# MODIFIED: Helper function to update target value function network
def update_target_value(vf, vf_target, tau):
    for target_param, param in zip(vf_target.parameters(), vf.parameters()):
        target_param.data.copy_((1. - tau) * target_param.data + tau * param.data)

# Sync weights of the QValues
# Setting tau to 1.0 is equivalent to Hard Update
update_target_value(qf, qf_target, 1.0)
update_target_value(pg, pg_target, 1.0)

q_optimizer = optim.Adam(list(qf.parameters()),
    lr=args.learning_rate)
p_optimizer = optim.Adam(list(pg.parameters()),
    lr=args.learning_rate)

mse_loss_fn = nn.MSELoss()

# Helper function to evaluate agent determinisitically
def test_agent(env, policy, eval_episodes=1):
    returns = []
    lengths = []

    for eval_ep in range(eval_episodes):
        ret = 0.
        done = False
        t = 0

        obs = np.array(env.reset())

        while not done:
            with torch.no_grad():
                # TODO: Revised sampling for determinsitc policy
                action = pg.forward([obs]).tolist()[0]

            obs, rew, done, _ = env.step(action)
            obs = np.array(obs)
            ret += rew
            t += 1
        # TODO: Break if max episode length is breached

        returns.append(ret)
        lengths.append(t)

    return returns, lengths

# TRY NOT TO MODIFY: start the game
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

if not args.notb:
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, tensorboard=True, config=vars(args), name=experiment_name)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    wandb.save(os.path.abspath(__file__))

global_step = 0
global_iter = 0
global_episode = 0

while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    done = False

    # MODIFIED: Keeping track of train episode returns and lengths
    train_episode_return = 0.
    train_episode_length = 0

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs = next_obs.copy()

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            # TODO: Noising process ?
            if global_step > args.start_steps:
                action = pg.forward([obs]).tolist()[0]
                if args.noise_type == "ou":
                    # Ourstein Uhlenbeck noise from baseline
                    action += ou_act_noise()
                elif args.noise_type == "naive":
                    action += args.noise_std * np.random.randn(output_shape) # From OpenAI SpinUp
                else:
                    raise NotImplementedError

                action = np.clip( action, -act_limit, act_limit)

            else:
                action = env.action_space.sample()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rew, done, _ = env.step(action)
        next_obs = np.array(next_obs)

        buffer.add_transition(obs, action, rew, done, next_obs)

        # Keeping track of train episode returns
        train_episode_return += rew
        train_episode_length += 1

        if done:
            # ALGO LOGIC: training.
            if global_step > args.start_steps:
                for iter in range( train_episode_length):
                    global_iter += 1

                    observation_batch, action_batch, reward_batch, \
                    terminals_batch, next_observation_batch = buffer.sample(args.batch_size)

                    # Q function losses and update
                    with torch.no_grad():
                        next_mus = pg_target.forward( next_observation_batch)
                        q_backup = torch.Tensor(reward_batch).to(device) + \
                            (1 - torch.Tensor(terminals_batch).to(device)) * args.gamma * \
                            qf_target.forward( next_observation_batch, next_mus).view(-1)

                    q_values = qf.forward( observation_batch, action_batch).view(-1)
                    q_loss = mse_loss_fn( q_values, q_backup)

                    q_optimizer.zero_grad()
                    q_loss.backward()
                    q_optimizer.step()

                    # Policy loss and update
                    resampled_mus = pg.forward( observation_batch)
                    q_mus = qf.forward( observation_batch, resampled_mus).view(-1)
                    policy_loss = - q_mus.mean()

                    p_optimizer.zero_grad()
                    policy_loss.backward()
                    p_optimizer.step()

                    if global_iter > 0 and global_iter % args.target_update_interval == 0:
                        update_target_value(qf, qf_target, args.tau)
                        update_target_value(pg, pg_target, args.tau)

                    if global_iter > 0 and global_iter % args.episode_length == 0:
                        # Evaulating in deterministic mode after one episode
                        eval_returns, eval_ep_lengths = test_agent(env, pg, 3)
                        eval_return_mean = np.mean(eval_returns)
                        eval_ep_length_mean = np.mean(eval_ep_lengths)

                        if not args.notb:
                            writer.add_scalar("train/q_loss", q_loss.item(), global_iter)
                            writer.add_scalar("train/policy_loss", policy_loss.item(), global_iter)

                            writer.add_scalar("eval/episode_return", eval_return_mean, global_iter)
                            writer.add_scalar("eval/episode_length", eval_ep_length_mean, global_iter)

                        print("Iter %d: PLoss: %.6f -- QLoss: %.6f -- Train Return: %.6f -- Test Mean Ret.: %.6f"
                            % (global_iter, policy_loss.item(), q_loss.item(), train_episode_return, eval_return_mean))

                if not args.notb:
                    writer.add_scalar("eval/train_episode_return", train_episode_return, global_iter)
                    writer.add_scalar("eval/train_episode_length", train_episode_length, global_iter)

            train_episode_return = 0.
            train_episode_length = 0

            global_episode += 1

            # Need to reset OrnsteinUhlenbeckActionNoise after each episode apparently
            if args.noise_type == "ou":
                ou_act_noise.reset()

            break;


if not args.notb:
    writer.close()
