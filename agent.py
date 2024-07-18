import torch
from networks import Actor, Critic, CNNActor, CNNCritic
from memory import ReplayBuffer
import numpy as np



class DiscretePPOAgent:
    def __init__(
        self,
        env_name,
        input_dims,
        n_actions,
        gamma=0.99,
        alpha=1e-4,
        gae_lambda=0.95,
        policy_clip=0.1,
        batch_size=64,
        n_epochs=10,
        max_grad_norm=0.5,
        entropy_coefficient=1e-2,
    ):
        self.env_name = env_name.split("/")[-1]
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm

        if "ALE/" in env_name or "NoFrameskip" in env_name:
            print("Learning from pixels with CNN Policy")
            self.actor = CNNActor(
                input_dims,
                n_actions,
                alpha,
                chkpt_dir=f"weights/{self.env_name}_actor.pt",
            )
            self.critic = CNNCritic(
                input_dims, 1e-3, chkpt_dir=f"weights/{self.env_name}_critic.pt"
            )
        else:
            print("Learning from features with MLP Policy")
            self.actor = Actor(
                input_dims,
                n_actions,
                alpha,
                chkpt_dir=f"weights/{self.env_name}_actor.pt",
            )
            self.critic = Critic(
                input_dims, alpha, chkpt_dir=f"weights/{self.env_name}_critic.pt"
            )

        self.memory = ReplayBuffer(batch_size)

    def remember(self, state, state_, action, probs, reward, done):
        self.memory.store_transition(state, state_, action, probs, reward, done)

    def save_checkpoints(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_checkpoints(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.actor.device).unsqueeze(0)

            dist = self.actor(state)
            action = dist.sample()
            prob = dist.log_prob(action)
            value = self.critic(state)

        return (
            action.cpu().numpy().flatten().item(),
            prob.cpu().numpy().flatten().item(),
            value.cpu().numpy().flatten().item(),
        )

    def evaluate_surrogate(self, state, action):
        dist = self.actor(state)
        prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state)
        return prob, entropy, value

    def learn(self):
        state_arr, value_arr, action_arr, prob_arr, reward_arr, dones_arr = (
            self.memory.sample()
        )

        returns = []
        discounted_return = 0
        for reward, done in zip(reversed(reward_arr), reversed(dones_arr)):
            if done:
                discounted_return = 0
            discounted_return = reward + (self.gamma * discounted_return)
            returns.insert(0, discounted_return)
        returns = torch.FloatTensor(np.array(returns)).to(self.critic.device)

        state_arr = torch.FloatTensor(state_arr).to(self.critic.device)
        action_arr = torch.FloatTensor(action_arr).to(self.critic.device)
        prob_arr = torch.FloatTensor(prob_arr).to(self.critic.device)
        value_arr = torch.FloatTensor(value_arr).to(self.critic.device)
        dones_arr = torch.BoolTensor(dones_arr).to(self.critic.device)
        reward_arr = torch.FloatTensor(reward_arr).to(self.critic.device)

        advantages_arr = reward_arr - value_arr
        # print("")
        # print("rewards:", reward_arr.shape)
        # print("values:", value_arr.shape)
        # print("advantages:", advantages_arr.shape)

        for _ in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                states = state_arr[batch]
                actions = action_arr[batch]
                old_probs = prob_arr[batch]
                advantages = advantages_arr[batch]
                new_probs, values, entropy = self.evaluate_surrogate(states, actions)

                prob_ratio = torch.exp(new_probs - old_probs)
                # print("old_probs:", old_probs.shape)
                # print("new_probs:", new_probs.shape)
                # print("prov_ratio:", prob_ratio.shape)
                # print("advantages:", advantages.shape)

                # surrogate loss
                weighted_probs = advantages * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantages
                )

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                # print("actor_loss:", actor_loss.shape)
                # print("entropy:", entropy.shape)
                actor_loss -= self.entropy_coefficient * entropy.squeeze()

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss = actor_loss.mean() + 0.5 * (values - returns[batch]).pow(2).mean()
                loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
