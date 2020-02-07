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
from gym.wrappers import TimeLimit, Monitor
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="HopperBulletEnv-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=100000,
                       help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                       help='whether to set `torch.backends.cudnn.deterministic=True`')
    parser.add_argument('--cuda', type=bool, default=True,
                       help='whether to use CUDA whenever possible')
    parser.add_argument('--prod-mode', type=bool, default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=bool, default=False,
                       help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                       help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help="the entity (team) of wandb's project")
    # MODIFIED: Disabled tensorboard logging
    parser.add_argument('--notb', action='store_true',
                help='Used to disable Tensorboard logging for algorithmic debugs')

    # Algorithm specific arguments
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--vf-coef', type=float, default=0.25,
                       help="value function's coefficient the loss function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help="policy entropy's coefficient the loss function")
    parser.add_argument('--clip-coef', type=float, default=0.2,
                       help="the surrogate clipping coefficient")
    # MODIFIED: Slight variant
    parser.add_argument('--epoch-length', type=int, default=1000,
                        help='the maximum length of each epoch AKA how many samples before updates')
    parser.add_argument('--update-epochs', type=int, default=100,
                        help="the K iterations to update the policy")
    parser.add_argument('--nokl', action='store_true',
                        help="Disable KL-based early stopping of policy updates")
    parser.add_argument('--target-kl', type=float, default=0.015)
    parser.add_argument('--value-lr', type=float, default=1e-3)
    parser.add_argument('--policy-lr', type=float, default=3e-4)
    parser.add_argument('--lam', type=float, default=0.97,
                        help='GAE Lambda coef.')
    # MODIFIED: State and Reward norm experiment
    parser.add_argument('--norm-obs', action='store_true',
                        help="Do we normalize the env's observations ?")
    parser.add_argument('--norm-rewards', action='store_true',
                        help="Do we normalize the env's rewards ?")
    parser.add_argument('--norm-returns', action='store_true',
                        help="Do we normalize the env's returns ?")
    parser.add_argument('--obs-clip', type=float, default=99999.0,
                        help="Max value the reward are clipped with")
    parser.add_argument('--rew-clip', type=float, default=99999.0,
                        help="Max value the reward are clipped with")
    # MODIFIED: Toggles for ADAM LR Annealing
    parser.add_argument('--anneal-lr', action='store_true',
                        help="Toggle learning rate annealing for policy and value")

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
# TMP MODIFIED
if not args.notb:
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    wandb.save(os.path.abspath(__file__))

# TRY NOT TO MODIFY: seeding
device = torch.device('cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
# MODIFIED: Env wrapping for state / reward normalization support
# env = gym.make(args.gym_id) # THis is defered to the custom env class, alter
# Filter and Running Stats
from cleanrl.experiments.torch_utils import RunningStat, ZFilter, Identity, StateWithTime, RewardFilter
# Custom Env with Normalization support and such
class CustomEnv(object):
    '''
    A wrapper around the OpenAI gym environment that adds support for the following:
    - Rewards normalization
    - State normalization
    - Adding timestep as a feature with a particular horizon T
    Also provides utility functions/properties for:
    - Whether the env is discrete or continuous
    - Size of feature space
    - Size of action space
    Provides the same API (init, step, reset) as the OpenAI gym
    '''
    def __init__(self, game, add_t_with_horizon=None):
        self.env = gym.make(game)

        self.env.seed(args.seed)
        self.env.action_space.seed(args.seed)
        self.env.observation_space.seed(args.seed)

        # Environment type
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Box

        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1 # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]

        # Number of features
        assert len(self.env.observation_space.shape) == 1
        self.num_features = self.env.reset().shape[0]

        # Support for state normalization or using time as a feature
        self.state_filter = Identity()
        if args.norm_obs:
            self.state_filter = ZFilter(self.state_filter, shape=[self.num_features], \
                                            clip=args.obs_clip)
        if add_t_with_horizon is not None:
            self.state_filter = StateWithTime(self.state_filter, horizon=add_t_with_horizon)

        # Support for rewards normalization
        self.reward_filter = Identity()
        if args.norm_rewards:
            self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=args.rew_clip)
        if args.norm_returns:
            self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=args.gamma, clip=args.rew_clip)

        # Running total reward (set to 0.0 at resets)
        self.total_true_reward = 0.0

    def reset(self):
        # Reset the state, and the running total reward
        start_state = self.env.reset()
        self.total_true_reward = 0.0
        self.counter = 0.0
        self.state_filter.reset()
        return self.state_filter(start_state, reset=True)

    def step(self, action):
        state, reward, is_done, info = self.env.step(action)
        state = self.state_filter(state)
        self.total_true_reward += reward
        self.counter += 1
        _reward = self.reward_filter(reward)
        if is_done:
            info['done'] = (self.counter, self.total_true_reward)
        return state, _reward, is_done, info

env = CustomEnv( args.gym_id)

input_shape, preprocess_obs_fn = preprocess_obs_space(env.env.observation_space, device)
output_shape = preprocess_ac_space(env.env.action_space)

# respect the default timelimit
if int(args.episode_length):
    if not isinstance(env, TimeLimit):
        env = TimeLimit(env, int(args.episode_length))
    else:
        env._max_episode_steps = int(args.episode_length)
else:
    args.episode_length = env._max_episode_steps if isinstance(env, TimeLimit) else 1000
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# ALGO LOGIC: initialize agent here:
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, output_shape)
        self.logstd = nn.Parameter(torch.zeros(1, output_shape))

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_mean = self.mean(x)
        action_logstd = self.logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def get_action(self, x):
        mean, logstd = self.forward(x)
        std = torch.exp(logstd)
        probs = Normal(mean, std)
        action = probs.sample()
        return action, probs.log_prob(action).sum(1)

    # MODIFIED: Get Determinsitic action
    def get_det_action( self, x):
        mean, _ = self.forward(x)
        logstd = torch.zeros_like( mean).to(device)
        std = torch.exp(logstd)
        probs = Normal(mean, std)
        action = probs.sample()
        return action

    def get_logproba(self, x, actions):
        action_mean, action_logstd = self.forward(x)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        return dist.log_prob(actions).sum(1)

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

pg = Policy().to(device)
vf = Value().to(device)

# MODIFIED: Buffer for Episode / Epoch data
import collections
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def put_episode( self, episode_data):
        for obs, act, logp, ret, adv in episode_data:
            self.put((obs, act, logp, ret, adv))

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        obs_lst, act_lst, logp_lst, ret_lst, adv_lst = [], [], [], [], []

        for transition in mini_batch:
            obs, act, logp, ret, adv = transition
            obs_lst.append(obs)
            act_lst.append(act)
            logp_lst.append(logp)
            ret_lst.append(ret)
            adv_lst.append(adv)

        # NOTE: Do not Tensor preprocess observations as it is done in the policy / values function
        return obs_lst, \
               torch.tensor( act_lst).to(device), \
               torch.tensor(logp_lst).to(device), \
               torch.tensor(ret_lst).to(device), \
               torch.tensor(adv_lst).to(device)

    def size(self):
        return len(self.buffer)

buffer = ReplayBuffer(args.epoch_length)

# MODIFIED: Separate optimizer and learning rates
pg_optimizer = optim.Adam( list(pg.parameters()), lr=args.policy_lr)
v_optimizer = optim.Adam( list(vf.parameters()), lr=args.value_lr)

# MODIFIED: Adam learning rate annealing scheduler
if args.anneal_lr:
    anneal_fn = lambda f: 1-f / args.total_timesteps
    pg_lr_scheduler = optim.lr_scheduler.LambdaLR( pg_optimizer, lr_lambda=anneal_fn)
    vf_lr_scheduler = optim.lr_scheduler.LambdaLR( v_optimizer, lr_lambda=anneal_fn)

loss_fn = nn.MSELoss()

# TRY NOT TO MODIFY: start the game
global_step = 0
# MODIFIED: Sampler function
# Shameless import of RLLAB's magic
import scipy.signal
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
def sample():
    global global_step

    obs = env.reset()
    done = False

    # Temporary storage for episode data
    ep_obs, ep_acts, ep_logps, ep_rews, ep_vals = [], [], [], [], []

    # Some sampling stats
    ep_count = 0
    ep_lengths = []
    ep_Returns = [] # Keep track of the undiscounted return for each sampled episode

    for step in range( args.epoch_length):
        global_step += 1
        with torch.no_grad():
            action, action_logp = pg.get_action( [obs])
            action = action.numpy()[0]
            action_logp = action_logp.numpy()[0]

            v_obs =  vf.forward( [obs]).numpy()[0][0]

        # for i in range( len(obs)):
        #     writer.add_scalar( "debug/observation_%d" % i, obs[i], global_step)


        next_obs, rew, done, _ = env.step( action)

        # DEBUG: Comparing normalized vs unormalized rewards
        # if not args.notb:
        #     writer.add_scalar( "debug/reward", rew, global_step)

        ep_obs.append( obs)
        ep_acts.append( action)
        ep_rews.append( rew)
        ep_logps.append( action_logp)
        ep_vals.append( v_obs)

        obs = next_obs

        if done:
            # Logging some stats
            ep_count += 1
            ep_lengths.append( step)
            ep_Returns.append( np.sum( ep_rews))

            ep_returns = discount_cumsum( ep_rews, args.gamma)
            # Hack GAE computation
            ep_rews = np.array( ep_rews)
            ep_vals = np.array( ep_vals)
            deltas = ep_rews[:-1] + args.gamma * ep_vals[1:] - ep_vals[:-1]
            ep_vals = discount_cumsum(deltas, args.gamma * args.lam)

            buffer.put_episode( zip( ep_obs, ep_acts, ep_logps, ep_returns, ep_vals))

            # Cleanup the tmp holders for fresh episode data
            ep_obs, ep_acts, ep_logps, ep_rews, ep_vals = [], [], [], [], []

            obs = env.reset()
            done = False

        # Reached sampling limit, will cut the episode and use vf to bootstrap
        if step == (args.episode_length - 1) and len(ep_obs) > 0:
            # Logging some stats
            ep_count += 1
            ep_lengths.append( step)
            ep_Returns.append( np.sum( ep_rews))

            # NOTE: Last condition: if the episode finished right at the sampling limit, skip
            with torch.no_grad():
                v_obs = vf.forward( [obs]).numpy()[0][0]

            ep_rews[-1] = v_obs # Boostrapping reward

            ep_returns = discount_cumsum( ep_rews, args.gamma)
            # Hack GAE computation
            ep_rews = np.array( ep_rews)
            ep_vals = np.array( ep_vals)
            deltas = ep_rews[:-1] + args.gamma * ep_vals[1:] - ep_vals[:-1]
            ep_vals = discount_cumsum(deltas, args.gamma * args.lam)

            buffer.put_episode( zip( ep_obs, ep_acts, ep_logps, ep_returns, ep_vals))

    sampling_stats = {
        "ep_count": ep_count,
        "train_episode_length": np.mean( ep_lengths),
        "train_episode_return": np.mean( ep_Returns)
    }

    return sampling_stats

while global_step < args.total_timesteps:
    # Will sample args.episode_length transitions, which we consider as an epoch
    sampling_stats = sample()

    # Sampling data from buffer, all of it
    obs_batch, act_batch, old_logp_batch, return_batch, adv = buffer.sample( buffer.size())

    # Train policy
    for i_epoch_pi in range( args.update_epochs):
        # Resample logps
        logp_a = pg.get_logproba( obs_batch, act_batch)
        ratio = (logp_a - old_logp_batch).exp()

        clip_adv = torch.where( adv > 0,
                                (1.+args.clip_coef) * adv,
                                (1.-args.clip_coef) * adv)
        # Q: There is no gradient for the clipped adv.
        policy_loss = - torch.min( ratio * adv, clip_adv).mean()

        pg_optimizer.zero_grad()
        policy_loss.backward()
        pg_optimizer.step()

        if not args.nokl:
            approx_kl = (old_logp_batch - logp_a).mean()
            if approx_kl > args.target_kl:
                break

    for i_epoch in range( args.update_epochs):
        v_obs_batch = vf.forward( obs_batch).view(-1)
        v_loss = loss_fn( return_batch, v_obs_batch)

        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()

    # Adam Learning Rate Annealing
    # TODO: Discuss wheter the annealing is top be done after each update or once the epoch is done
    # THe following is the easiest: After the epoch.
    # However, given that we only pass the args.total_timesteps to the annealing function, Torch
    # probably tracks how many times the {pg,v}_optimizer's step function is called and then anneals
    # according.
    if args.anneal_lr:
        pg_lr_scheduler.step()
        vf_lr_scheduler.step()

    if not args.notb:
        writer.add_scalar("charts/episode_reward",
            sampling_stats["train_episode_return"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)

        # MODIFIED: After how many iters did the policy udate stop ?
        if not args.nokl:
            writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)

        # MODIFIED: Also logging the learning rate as the training progesses (debug purposes)
        # Debug done: Anneals indeed.
        # if args.anneal_lr:
        #     writer.add_scalar("debug/policy_lr", pg_lr_scheduler.get_lr()[0], global_step)
        #     writer.add_scalar("debug/value_lr", vf_lr_scheduler.get_lr()[0], global_step)

        writer.add_scalar("debug/episode_count", sampling_stats["ep_count"], global_step)
        writer.add_scalar("debug/mean_episode_length", sampling_stats["ep_count"], global_step)

    print( "Step %d -- PLoss: %.6f -- VLoss: %.6f -- Train Mean Return: %.6f" % (global_step,
        policy_loss.item(), v_loss.item(), sampling_stats["train_episode_return"]))
env.close()
writer.close()
