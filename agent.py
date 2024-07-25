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
        alpha=3e-4,
        gae_lambda=0.95,
        policy_clip=0.1,
        batch_size=64,
        n_epochs=5,
        max_grad_norm=0.5,
        entropy_coefficient=0.01,
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
                input_dims, alpha, chkpt_dir=f"weights/{self.env_name}_critic.pt"
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

    def remember(self, state, value, action, probs, reward, done):
        self.memory.store_transition(state, value, action, probs, reward, done)

    def save_checkpoints(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_checkpoints(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state, action=None):
        state = torch.FloatTensor(state).to(self.actor.device).unsqueeze(0)

        dist = self.actor(state)
        if action == None:
            action = dist.sample()
        prob = dist.log_prob(action)
        value = self.critic(state)
        entropy = dist.entropy()

        return (
            action.cpu().numpy().flatten().item(),
            prob.cpu().numpy().flatten().item(),
            value.cpu().numpy().flatten().item(),
            entropy.cpu().numpy().flatten().item()
        )

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

        returns_arr = torch.FloatTensor(np.array(returns)).to(self.critic.device)
        state_arr = torch.FloatTensor(state_arr).to(self.critic.device)
        action_arr = torch.FloatTensor(action_arr).to(self.critic.device)
        prob_arr = torch.FloatTensor(prob_arr).to(self.critic.device)
        value_arr = torch.FloatTensor(value_arr).to(self.critic.device)
        dones_arr = torch.BoolTensor(dones_arr).to(self.critic.device)
        reward_arr = torch.FloatTensor(reward_arr).to(self.critic.device)

        advantages_arr = reward_arr - value_arr

        for _ in range(self.n_epochs):
            batches = self.memory.generate_batches()
            
            for batch in batches:
                states = state_arr[batch]
                actions = action_arr[batch]
                old_values = value_arr[batch]
                old_probs = prob_arr[batch]
                advantages = advantages_arr[batch]
                returns = returns_arr[batch]

                advantages = (advantages - advantages.mean())/(advantages.std() + 1e-12)
                _, new_probs, new_values, entropy = self.choose_action(states, actions)
                
                logratio = new_probs - old_probs
                ratio = logratio.exp()

                weighted_probs = -advantages * ratio
                weighted_clipped_probs = -(
                    torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantages
                )

                actor_loss = torch.max(weighted_probs, weighted_clipped_probs).mean()
                
                entropy_loss = self.entropy_coefficient * entropy.mean()

                unclipped_critic_loss = (new_values - returns).pow(2)
                clipped_critic_loss = old_values + torch.clamp(new_values - old_values, -self.policy_clip, self.policy_clip)
                clipped_critic_loss = (clipped_critic_loss - returns)**2
                critic_loss = 0.5 * torch.max(unclipped_critic_loss, clipped_critic_loss).mean()


                loss = actor_loss - entropy_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
