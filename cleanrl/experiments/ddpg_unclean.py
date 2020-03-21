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
    parser.add_argument('--prod-mode', default=False, action="store_true",
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
    parser.add_argument('--target-update-interval', type=int, default=1,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--batch-size', type=int, default=128,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--start-steps', type=int, default=int(1e4),
                        help='initial random exploration step count')

    # TODO: Add Parameter Noising support ~ https://arxiv.org/abs/1706.01905
    parser.add_argument('--noise-type', type=str, choices=["normal", "ou", "param", "adapt-param"], default="normal",
                        help='type of noise to be used when sampling exploratiry actions')
    parser.add_argument('--noise-mean', type=float, default=0.0,
                        help="mean for the noise process")
    parser.add_argument('--noise-std', type=float, default=0.2,
                        help="standard deviation of the noise process")
    parser.add_argument('--param-noise-adapt-interval', type=int, default=50,
                        help="Defines the updated interval we adapt the std of the AdaptiveParamNoiseSpec")
    # Experiments:
    # TODO: Discuss whether to set it as a default or not. param space noise paper says that's What
    # the original paper did. P12 A.2
    parser.add_argument('--layer-norm', default=False, action="store_true",
                        help='Toggles layers norm in the actor and critic\s networks')

    ## Adding noise to the target mus to, as in TD3, but only one Q network
    parser.add_argument('--target-mus-noise', action="store_true",
        help='If passed, will add some noise to the target next mus')

    # TODO: Remove this trash latter
    # Neural Network Parametrization
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[256,128,80])

    parser.add_argument('--notb', action='store_true',
        help='No Tensorboard logging')

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

# MODIFIED: Added noise functions for exploration, copied from Baselines
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class NormalActionNoise(object):
    def __init__(self, mu, sigma, shape=1):
        self.mu = mu
        self.sigma = sigma
        self.shape = shape

    def __call__(self):
        return np.random.normal(self.mu, self.sigma, self.shape)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)

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

if args.noise_type == "normal":
    noise_process = lambda: NormalActionNoise( np.ones( output_shape) * args.noise_mean, np.ones(output_shape) * float(args.noise_std))()
elif args.noise_type == "ou":
    noise_process = OrnsteinUhlenbeckActionNoise( np.ones(output_shape) * args.noise_mean, np.ones(output_shape) * float(args.noise_std))
elif args.noise_type == "param":
    # Naive version: just sample a single noise value and add it to all the weights in the network
    noise_process = lambda: NormalActionNoise(np.ones(1) * args.noise_mean, np.ones(1) * args.noise_std)()
elif args.noise_type == "adapt-param":
    # x is the shape
    adaptive_param_noise = AdaptiveParamNoiseSpec( args.noise_std, args.noise_std)
    noise_process = lambda shape: NormalActionNoise(np.ones(1) * args.noise_mean,
        np.ones(1) * adaptive_param_noise.current_stddev, shape=shape)()
else:
    raise NotImplementedError
# THIS is a function that depends on the type of noise passed earlier.
if args.noise_type == "adapt-param":
    # Quick workaround as for adapt-param, we need noise with a specific shape !
    noise_fn = lambda shape: noise_process(shape)
else:
    noise_fn = lambda: noise_process()

# Determinsitic policy
# TODO: Refactor Policy and QFunction the CleanRL way
class Policy(nn.Module):
    def __init__(self):
        # Custom
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        current_dim = input_shape
        for hsize in args.hidden_sizes:
            self.layers.append(nn.Linear(current_dim, hsize))
            if args.layer_norm:
                self.layer_norms.append(nn.LayerNorm(hsize))
            current_dim = hsize

        # TODO: Add squashing like mechanism for action bound inforcing
        self.fc_mu = nn.Linear(args.hidden_sizes[-1], output_shape)
        if args.layer_norm:
            self.mu_ln = nn.LayerNorm(output_shape) # Probably not that helpful since passed to tanh (?)
        self.tanh_mu = nn.Tanh()

    def forward(self, x):
        # # Workaround Seg. Fault when passing tensor to Q Function
        if not isinstance(x, torch.Tensor):
            x = preprocess_obs_fn(x)

        for layer_idx, layer in enumerate(self.layers):
            x = F.relu(layer(x))

            if args.layer_norm:
                x = self.layer_norms[layer_idx](x)

        mus = self.fc_mu(x)

        if args.layer_norm:
            mus = self.mu_ln(mus)

        return self.tanh_mu( mus)

class QValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        current_dim = input_shape + output_shape
        for hsize in args.hidden_sizes:
            self.layers.append(nn.Linear(current_dim, hsize))
            if args.layer_norm:
                self.layer_norms.append(nn.LayerNorm(hsize))
            current_dim = hsize

        self.layers.append(nn.Linear(args.hidden_sizes[-1], 1))

    def forward(self, x, a):
        # Workaround Seg. Fault when passing tensor to Q Function
        if not isinstance(x, torch.Tensor):
            x = preprocess_obs_fn(x)

        if not isinstance(a, torch.Tensor):
            a = preprocess_obs_fn(a)

        x = torch.cat([x,a], 1)

        for layer_idx, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            if args.layer_norm:
                x = self.layer_norms[layer_idx](x)

        return self.layers[-1](x)

buffer = SimpleReplayBuffer(env.observation_space, env.action_space, args.buffer_size, args.batch_size)
buffer.set_seed(args.seed)

# Defining agent's policy and corresponding target
pg = Policy().to(device)
pg_target = Policy().to(device)

# Setup all the necessary to enable parameter space noising method,
# with both naive and adaptive version
if args.noise_type == "param":
    pg_exploration = Policy().to(device)
elif args.noise_type == "adapt-param":
    pg_exploration = Policy().to(device)
    pg_adaptive_noise = Policy().to(device)

    if args.target_mus_noise:
        pg_target_exploration = Policy().to(device)

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

# TODO: Separate learning rate for policy and value func.
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

    # SPECIAL: In case we use parameter space noise, we need to copy the current
    # actor network and and noise up its weights, then use it to sample
    # TODO: Also, in case we still using action_space.sample(), do not copy weights etc...
    if args.noise_type == "param":
        p_optimizer.zero_grad()# Just for safety
        pg_exploration.load_state_dict( pg.state_dict()) # Copy current weight to exploratory policy

        if args.target_mus_noise:
            pg_target_exploration.load_state_dict( pg_target.state_dict())

        # NAIVE version: No adaptive parameter noise as in the paper, just adding
        # the same noise to all the weights, sampled from a Gaussian Noise
        # NOITE: When using Naive method for param noise, too big noise-std will results in action always being [1.] * act_shape
        noise = noise_fn()[0]

        with torch.no_grad():
            for param in pg_exploration.parameters():
                param += noise

            if args.target_mus_noise:
                for param in pg_target_exploration.parameters():
                    param += noise

    elif args.noise_type == "adapt-param":
        # TODO: Generate a random noise for each of the individual weights (bias included it seems)
        p_optimizer.zero_grad()# Just for safety
        pg_exploration.load_state_dict( pg.state_dict()) # Copy current weight to exploratory policy

        if args.target_mus_noise:
            pg_target_exploration.load_state_dict( pg_target.state_dict())

        with torch.no_grad():
            for param in pg_exploration.parameters():
             # Generate noise corresponding to the shape of the layer and add it
                # TODO: More rigorous debug. Can we actually see the matrix + matrix addition of the noise ?
                param.add_(torch.Tensor(noise_fn(param.shape)).to(device))

            if args.target_mus_noise:
                for param in pg_target_exploration.parameters():
                 # Generate noise corresponding to the shape of the layer and add it
                    # TODO: More rigorous debug. Can we actually see the matrix + matrix addition of the noise ?
                    param.add_(torch.Tensor(noise_fn(param.shape)).to(device))

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs = next_obs.copy()

        # ALGO LOGIC: put action logic here
        if global_step > args.start_steps:

            with torch.no_grad():
                if args.noise_type in ["ou", "normal"]:
                    action = pg.forward([obs]).tolist()[0]

                    # NOTE: This is already preconfigured to support a specific noise function
                    action += noise_fn()
                elif args.noise_type in ["param","adapt-param"]:
                    action = pg_exploration.forward([obs]).tolist()[0]
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
                        if args.target_mus_noise:
                            if args.noise_type in ["normal", "ou"]:
                                next_mus = pg_target(next_observation_batch)
                                next_mus += torch.Tensor(noise_fn()).to(device)
                                next_mus.clamp_( -act_limit, act_limit)
                            elif args.noise_type in ["param","adapt-param"]:
                                next_mus = pg_target_exploration(next_observation_batch)
                        else:
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

                    # Updaing the std of the AdaptiveParamNoiseSpec
                    if args.noise_type == "adapt-param":
                        if global_iter > 0 and global_iter % args.param_noise_adapt_interval == 0:
                            # First, perturb the current network again
                            p_optimizer.zero_grad()# Just for safety
                            pg_adaptive_noise.load_state_dict( pg.state_dict())

                            with torch.no_grad():
                                for param in pg_adaptive_noise.parameters():
                                 # Generate noise corresponding to the shape of the layer and add it
                                    # TODO: More rigorous debug. Can we actually see the matrix + matrix addition of the noise ?
                                    param.add_(torch.Tensor(noise_fn(param.shape)).to(device))

                            distance = mse_loss_fn( pg(observation_batch), pg_adaptive_noise(observation_batch))

                            adaptive_param_noise.adapt(distance)

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
                noise_process.reset()
                noise_fn = lambda: noise_process()

            break;


if not args.notb:
    writer.close()
