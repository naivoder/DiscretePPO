import torch
from networks import Actor, Critic
from memory import ReplayBuffer


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
        self.actor = Actor(
            input_dims, n_actions, alpha, chkpt_dir=f"weights/{self.env_name}_actor.pt"
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
            probs = dist.log_prob(action)

        return (
            action.cpu().numpy().flatten().item(),
            probs.cpu().numpy().flatten().item(),
        )

    def calc_adv_and_returns(self, memories):
        states, new_states, rewards, dones = memories
        with torch.no_grad():
            values = self.critic(states)
            values_ = self.critic(new_states)
            deltas = rewards + self.gamma * values_ - values
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]
            for delta, done in zip(deltas[::-1], dones[::-1]):
                advantage = delta + self.gamma * self.gae_lambda * adv[-1] * (1 - done)
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]
            adv = torch.tensor(adv).float().unsqueeze(1).to(self.critic.device)
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def learn(self):
        state_arr, new_state_arr, action_arr, old_prob_arr, reward_arr, dones_arr = (
            self.memory.sample()
        )

        state_arr = torch.FloatTensor(state_arr).to(self.critic.device)
        action_arr = torch.FloatTensor(action_arr).to(self.critic.device)
        old_prob_arr = torch.FloatTensor(old_prob_arr).to(self.critic.device)
        new_state_arr = torch.FloatTensor(new_state_arr).to(self.critic.device)
        reward_arr = torch.FloatTensor(reward_arr).unsqueeze(1).to(self.critic.device)

        advantages, returns = self.calc_adv_and_returns(
            (state_arr, new_state_arr, reward_arr, dones_arr)
        )

        for _ in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]

                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                prob_ratio = torch.exp(
                    new_probs.sum(-1, keepdim=True) - old_probs.sum(-1, keepdim=True)
                )

                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantages[batch]
                )

                entropy = torch.mean(dist.entropy())
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                actor_loss -= self.entropy_coefficient * entropy

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor.optimizer.step()

                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[batch]).pow(2).mean()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic.optimizer.step()

        self.memory.clear_memory()
