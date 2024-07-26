import torch
from memory import ReplayBuffer
import numpy as np

class ActorCritic(torch.nn.Module):
    def __init__(self, input_dims, n_actions, alpha=1e-4, chkpt_dir="weights/network.pt"):
        super(ActorCritic, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.chkpt_dir = chkpt_dir

        self.conv1 = self._init_weights(torch.nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4))
        self.conv2 = self._init_weights(torch.nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = self._init_weights(torch.nn.Conv2d(64, 64, kernel_size=3, stride=1))

        self.fc1_input_dim = self._calculate_fc1_input_dim(input_dims)
        self.fc1 = self._init_weights(torch.nn.Linear(self.fc1_input_dim, 512))
        self.critic = self._init_weights(torch.nn.Linear(512, 1), std=1.0)
        self.actor = self._init_weights(torch.nn.Linear(512, n_actions), std=0.01)

        self.optimizer = torch.optim.AdamW(self.parameters(), alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _init_weights(self, layer, std=np.sqrt(2), scale=False):
        """taken from cleanrl implementation"""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, 0)
        return layer

    def _calculate_fc1_input_dim(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        x = torch.nn.functional.relu(self.conv1(dummy_input))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        return x.numel()  # count flattened elements

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        value = self.critic(x)
        x = torch.nn.functional.softmax(self.actor(x))
        action = torch.distributions.Categorical(logits=x)
        return action, value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_dir)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_dir))

class DiscretePPOAgent:
    def __init__(
        self,
        env_name,
        input_dims,
        n_actions,
        gamma=0.99,
        alpha=2.5e-4,
        gae_lambda=0.95,
        policy_clip=0.1,
        batch_size=64,
        n_epochs=5,
        max_grad_norm=0.5,
        entropy_coefficient=0.01,
        clip_value=True,
    ):
        self.env_name = env_name.split("/")[-1]
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.clip_value = clip_value

        self.network = ActorCritic(input_dims, n_actions, alpha, f"weights/{env_name}.pt")

        self.memory = ReplayBuffer(batch_size)

    def remember(self, state, value, action, probs, reward, done):
        self.memory.store_transition(state, value, action, probs, reward, done)

    def save_checkpoints(self):
        self.network.save_checkpoint()

    def load_checkpoints(self):
        self.network.load_checkpoint()

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.network.device).unsqueeze(0)
        dist, value = self.network(state)
        action = dist.sample()
        prob = dist.log_prob(action)
        return action.item(), prob.item(), value.item()

    def learn(self):
        state_arr, value_arr, action_arr, prob_arr, reward_arr, dones_arr = (
            self.memory.sample()
        )

        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                a_t += discount*(reward_arr[k] + self.gamma*value_arr[k+1]*\
                        (1-int(dones_arr[k])) - value_arr[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t
        
        advantage_arr = torch.tensor(advantage).to(self.network.device)
        state_arr = torch.FloatTensor(state_arr).to(self.network.device)
        action_arr = torch.FloatTensor(action_arr).to(self.network.device)
        prob_arr = torch.FloatTensor(prob_arr).to(self.network.device)
        value_arr = torch.FloatTensor(value_arr).to(self.network.device)
        dones_arr = torch.BoolTensor(dones_arr).to(self.network.device)
        reward_arr = torch.FloatTensor(reward_arr).to(self.network.device)

        for _ in range(self.n_epochs):
            batches = self.memory.generate_batches()
            
            for batch in batches:
                states = state_arr[batch]
                actions = action_arr[batch]
                old_values = value_arr[batch]
                old_probs = prob_arr[batch]
                advantages = advantage_arr[batch]
                
                dist, new_vals = self.network(states)
                new_values = new_vals.squeeze()

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantages * prob_ratio
                weighted_clipped_probs = advantages * torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
            
                actor_loss = torch.min(weighted_probs, weighted_clipped_probs).mean()
                actor_loss -= self.entropy_coefficient * dist.entropy().mean()

                returns = advantages + old_values

                if self.clip_value:
                    unclipped_critic_loss = (new_values - returns).pow(2)
                    clipped_critic_loss = old_values + torch.clamp(new_values - old_values, -self.policy_clip, self.policy_clip)
                    clipped_critic_loss = (clipped_critic_loss - returns)**2
                    critic_loss = 0.5 * torch.max(unclipped_critic_loss, clipped_critic_loss).mean()
                else:
                    critic_loss = (returns-new_values).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss

                self.network.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.network.optimizer.step()

        self.memory.clear_memory()
