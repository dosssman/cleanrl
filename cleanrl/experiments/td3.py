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
import collections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3')
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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', action='store_true', default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=int(1e5),
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    # THIS becomes useless in TD3
    # parser.add_argument('--target-update-interval', type=int, default=1,
    #                     help="the timesteps it takes to update the target network")
    parser.add_argument('--batch-size', type=int, default=100,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--start-steps', type=int, default=int(1e4),
                        help='initial random exploration step count')
    parser.add_argument('--update-delay', type=int, default=2,
                        help='Delay in timestep for actor and critic updates (?)')

    # TODO: Add Parameter Noising support ~ https://arxiv.org/abs/1706.01905
    # NOTE: This relates to the noise of the action during policy iteration
    parser.add_argument('--noise-type', type=str, choices=["ou", "normal"], default="normal",
                        help='type of noise to be used when sampling exploratiry actions')
    parser.add_argument('--noise-std', type=float, default=0.1,
                        help="standard deviation of the Normal dist for action noise sampling")
    # NOTE: This however, relates the noise applied to the next_mus used for critic updates via TD(0) Learning
    # By default, the other  suggest using Normal ( Gaussian) noise for the next_mus noise too
    parser.add_argument('--noise-std-target', type=float, default=0.2,
                        help="standard deviation of the Normal dist for action noise sampling")
    parser.add_argument('--noise-clip-coef', type=float, default=0.5,
                        help="standard deviation of the Normal dist for action noise sampling")

    # MODIFIED: Include deterministic evaluation of the agent
    parser.add_argument('--eval', action="store_true",
                        help='If passed, the script will evaluate the agent without the noise')
    parser.add_argument('--eval-interval', type=int, default=20,
                        help='Defines every X episodes the agent is evaluated fully determinisitically')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='How many episode do we evaluate the agent over ?')

    # TODO: Remove this trash once everything works as supposed
    parser.add_argument('--notb', action='store_true',
        help='No Tensorboard logging')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
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
    ou_act_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_shape), sigma=float(args.noise_std) * np.ones(output_shape))

# Determinsitic policy
class Policy(nn.Module):
    def __init__(self):
        # Custom
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, output_shape)

        self.tanh_mu = nn.Tanh() # To squash actions between -1.,1.

    def forward(self, x):
        x = preprocess_obs_fn(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh_mu(x)

        return x

class QValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_shape + output_shape, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, a):
        # GPU Seg. fault bug workaround
        if not isinstance(x, torch.Tensor):
            x = preprocess_obs_fn(x)

        if not isinstance(a, torch.Tensor):
            a = preprocess_obs_fn(a)

        x = torch.cat([x,a], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

buffer = ReplayBuffer(args.buffer_size)

# Defining agent's policy and corresponding target
pg = Policy().to(device)
pg_target = Policy().to(device)

qf1 = QValue().to(device)
qf2 = QValue().to(device)

qf1_target = QValue().to(device)
qf2_target = QValue().to(device)

# MODIFIED: Helper function to update target value function network
def update_target_value(vf, vf_target, tau):
    for target_param, param in zip(vf_target.parameters(), vf.parameters()):
        target_param.data.copy_((1. - tau) * target_param.data + tau * param.data)

# Sync weights of the QValues
# Setting tau to 1.0 is equivalent to Hard Update
update_target_value(qf1, qf1_target, 1.0)
update_target_value(qf2, qf1_target, 1.0)
update_target_value(pg, pg_target, 1.0)

# TODO: Separate learning rate for policy and value func.
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()),
    lr=args.learning_rate)
p_optimizer = optim.Adam(list(pg.parameters()),
    lr=args.learning_rate)

mse_loss_fn = nn.MSELoss()

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
        if global_step > args.start_steps:

            with torch.no_grad():
                if args.noise_type in ["ou", "normal"]:
                    action = pg.forward([obs]).tolist()[0]
                    if args.noise_type == "ou":
                        # Ourstein Uhlenbeck noise from baseline
                        action += ou_act_noise()
                    elif args.noise_type == "normal":
                        action += args.noise_std * np.random.randn(output_shape) # From OpenAI SpinUp
                elif args.noise_type == "param":
                    ### DANGER: Pondering how many time we must noise the weights between updates
                    # NAIVE Version: Sample the same epsilon and add it to every parameter
                    raise NotImplementedError

                    noise = args.noise_std * np.random.randn(1)[0]
                    for param in pg.parameters():
                        param += noise

                    action = pg.forward([obs]).tolist()[0]
                else:
                    raise NotImplementedError
                action = np.clip(action, -act_limit, act_limit)
        else:
            action = env.action_space.sample()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rew, done, _ = env.step(action)
        next_obs = np.array(next_obs)

        buffer.put((obs, action, rew, next_obs, done))

        # Keeping track of train episode returns
        train_episode_return += rew
        train_episode_length += 1

        if done:
            # ALGO LOGIC: training.
            if global_step > args.start_steps:
                for iter in range(train_episode_length): # As much updates as there was steps in the episode
                    global_iter += 1

                    observation_batch, action_batch, reward_batch, \
                        next_observation_batch, terminals_batch = buffer.sample(args.batch_size)

                    # Q function losses and update
                    with torch.no_grad():
                        next_mus = pg_target.forward(next_observation_batch)
                        # As per the paper, we also noisy up the next_mus
                        action = pg.forward([obs]).tolist()[0]

                        next_mus_noise = args.noise_std_target * np.random.randn(output_shape)

                        # Clipping the noises for next_mus and apply the noise to next_mus
                        np.clip(next_mus_noise, -args.noise_clip_coef, args.noise_clip_coef, next_mus_noise)
                        next_mus += torch.Tensor(next_mus_noise).to(device)

                        qf1_next_target = qf1_target.forward(next_observation_batch, next_mus).view(-1)
                        qf2_next_target = qf1_target.forward(next_observation_batch, next_mus).view(-1)
                        min_qf_next_target = torch.min(qf1_next_target,qf2_next_target)

                        q_backup = torch.Tensor(reward_batch).to(device) + \
                            (1 - torch.Tensor(terminals_batch).to(device)) * args.gamma * \
                            min_qf_next_target

                    qf1_values = qf1.forward(observation_batch, action_batch).view(-1)
                    qf2_values = qf2.forward(observation_batch, action_batch).view(-1)

                    qf1_loss = mse_loss_fn(qf1_values, q_backup)
                    qf2_loss = mse_loss_fn(qf2_values, q_backup)
                    qf_loss = (qf1_loss + qf2_loss).mean()

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()


                    # Updates the target network
                    # if global_iter > 0 and global_iter % args.target_update_interval == 0:
                    if global_iter > 0 and global_iter % args.update_delay == 0:
                        # Policy loss and update
                        resampled_mus = pg.forward(observation_batch)
                        # NOTE: The paper uses
                        qf1_mus = qf1(observation_batch, resampled_mus).view(-1)
                        qf2_mus = qf2(observation_batch, resampled_mus).view(-1)
                        min_qf_mus = torch.min(qf1_mus, qf2_mus)

                        policy_loss = - min_qf_mus.mean()

                        p_optimizer.zero_grad()
                        policy_loss.backward()
                        p_optimizer.step()

                        update_target_value(qf1, qf1_target, args.tau)
                        update_target_value(qf2, qf2_target, args.tau)
                        update_target_value(pg, pg_target, args.tau)

                    if global_iter > 0 and global_iter % 1000 == 0:
                        print("Iter %d: PLoss: %.6f -- QLoss: %.6f -- Train Return: %.6f"
                            % (global_iter, policy_loss.item(), qf_loss.item(), train_episode_return))

                if not args.notb:
                    # Logging training stats
                    writer.add_scalar("charts/episode_reward", train_episode_return, global_iter)
                    writer.add_scalar("charts/episode_length", train_episode_length, global_iter)

                    writer.add_scalar("train/q_loss", qf_loss.item(), global_iter)
                    writer.add_scalar("train/qf1_loss", qf1_loss.item(), global_iter)
                    writer.add_scalar("train/qf2_loss", qf1_loss.item(), global_iter)
                    writer.add_scalar("train/policy_loss", policy_loss.item(), global_iter)

                    # Deterministic evaluation loop
                    if args.eval and global_episode % args.eval_interval == 0:

                        eval_returns = []
                        eval_ep_lengths = []

                        for eval_ep in range(args.eval_episodes):
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

                            eval_returns.append(ret)
                            eval_ep_lengths.append(t)

                        eval_return_mean = np.mean(eval_returns)
                        eval_ep_length_mean = np.mean(eval_ep_lengths)

                        writer.add_scalar("charts/determinisitic_episode_reward", eval_return_mean, global_iter)
                        writer.add_scalar("charts/determinisitic_episode_length", eval_ep_length_mean, global_iter)

            train_episode_return = 0.
            train_episode_length = 0

            global_episode += 1

            # Need to reset OrnsteinUhlenbeckActionNoise after each episode apparently
            if args.noise_type == "ou":
                ou_act_noise.reset()

            break;

# TODO: Remove notb
if not args.notb:
    writer.close()
env.close()
