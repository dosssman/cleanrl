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
    parser.add_argument('--episode-length', type=int, default=1000,
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
                       help='the maximum length of each episode')
    parser.add_argument('--update-epochs', type=int, default=100,
                        help="the K epochs to update the policy")
    parser.add_argument('--target-kl', type=float, default=0.015)
    parser.add_argument('--value-lr', type=float, default=1e-3)
    parser.add_argument('--policy-lr', type=float, default=3e-4)
    parser.add_argument('--lam', type=float, default=0.97,
                        help='GAE Lambda coef.')
    parser.add_argument('--nokl', action='store_true',
                        help='If toggled, the policy updates will not be early stopped')

    # MODIFIED: Neural network init parametetization
    parser.add_argument('--weights-init', default="xavier", choices=["xavier", "normal", 'orthogonal'],
                        help='Selects the scheme used for weight initialization')
    # Because lazyness, bring back the hidden-size to programatically init the layers
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[256,256])

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
# respect the default timelimit
if int(args.episode_length):
    if not isinstance(env, TimeLimit):
        env = TimeLimit(env, int(args.episode_length))
    else:
        env._max_episode_steps = int(args.episode_length)
else:
    args.episode_length = env._max_episode_steps if isinstance(env, TimeLimit) else 200
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# ALGO LOGIC: initialize agent here:
# MODIFIED: Weight init function and helper from implementation-matters/code-for-paper
STD = 2**0.5
def orthogonal_init(tensor, gain=1):
    '''
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> orthogonal_init(w)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

def initialize_weights(mod, initialization_type, scale=STD):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.layers = nn.ModuleList()

        current_dim = input_shape
        for hsize in args.hidden_sizes:
            layer = nn.Linear( current_dim, hsize)
            initialize_weights( layer, args.weights_init)
            self.layers.append( layer)
            current_dim = hsize

        self.mean = nn.Linear(args.hidden_sizes[-1], output_shape)
        initialize_weights(self.mean, args.weights_init)
        # TODO: If we don't do this, does the params of this last layer end up in
        # the list when calling pg.parameters ( which is passed to the optimizer btw)
        self.layers.append( self.mean)

        self.logstd = nn.Parameter(torch.zeros(1, output_shape))
        # initialize_weights( logstd_layer, args.weights_init)
        # self.layers.append( self.logstd) # Can't add this to module list

    def forward(self, x):
        x = preprocess_obs_fn(x)
        # Forget the last 2 layers, which are mean and logstd
        for layer in self.layers[:-1]:
            x = torch.tanh( layer(x))

        action_mean = self.layers[-1](x) # -1 is the mean layer
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
        self.layers = nn.ModuleList()

        current_dim = input_shape
        for hsize in args.hidden_sizes + [1,]:
            layer = nn.Linear(current_dim, hsize)
            initialize_weights( layer, args.weights_init)
            self.layers.append( layer)
            current_dim = hsize
        # Note: The last layer will already be of dimension [H, 1]

    def forward(self, x):
        x = preprocess_obs_fn(x)
        for layer in self.layers[:-1]:
            x = torch.tanh( layer(x))

        # Final layer: No activation on this one !
        x = self.layers[-1]( x)

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

buffer = ReplayBuffer(args.episode_length)

# MODIFIED Helper for deterministic eval of agent
# def test_agent( env, policy, eval_episodes=1):
#     returns = []
#     lengths = []
#
#     for eval_ep in range( eval_episodes):
#         ret = 0.
#         done = False
#         t = 0
#
#         obs = np.array( env.reset())
#
#         while not done:
#             with torch.no_grad():
#                 action = pg.get_det_action([obs])
#                 action = action.numpy()[0]
#
#             obs, rew, done, _ = env.step( action)
#             obs = np.array( obs)
#             ret += rew
#             t += 1
#         # TODO: Break if max episode length is breached
#
#         returns.append( ret)
#         lengths.append( t)
#
#     return returns, lengths

# MODIFIED: Separate optimizer and learning rates
pg_optimizer = optim.Adam( list(pg.parameters()), lr=args.policy_lr)
v_optimizer = optim.Adam( list(vf.parameters()), lr=args.value_lr)
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

    for step in range( args.episode_length):
        global_step += 1
        with torch.no_grad():
            action, action_logp = pg.get_action( [obs])
            action = action.numpy()[0]
            action_logp = action_logp.numpy()[0]

            v_obs =  vf.forward( [obs]).numpy()[0][0]

        next_obs, rew, done, _ = env.step( action)

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
    obs_batch, act_batch, old_logp_batch, return_batch, adv_batch = buffer.sample( buffer.size())

    # Train policy
    for i_epoch_pi in range( args.update_epochs):
        # Resample logps
        logp_a = pg.get_logproba( obs_batch, act_batch)
        ratio = (logp_a - old_logp_batch).exp()

        adv = adv_batch
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
        # TODO: Gradient clip experiement
        v_optimizer.step()

    if not args.notb:
        writer.add_scalar("charts/episode_reward",
            sampling_stats["train_episode_return"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        # MODIFIED: After how many iters did the policy udate stop ?
        if not args.nokl:
            writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
        writer.add_scalar("debug/episode_count", sampling_stats["ep_count"], global_step)
        writer.add_scalar("debug/mean_episode_length", sampling_stats["ep_count"], global_step)

        # # MODIFIED: Evaluating agent in determinsitic mode
        # eval_returns, eval_lenghts = test_agent( env, pg, 3)
        # writer.add_scalar("debug/det_eval_return", np.mean( eval_returns), global_step)
        # writer.add_scalar("debug/det_eval_length", np.mean( eval_lenghts), global_step)


    print( "Step %d -- PLoss: %.6f -- VLoss: %.6f -- Train Mean Return: %.6f" % (global_step,
        policy_loss.item(), v_loss.item(), sampling_stats["train_episode_return"]))
env.close()
writer.close()
