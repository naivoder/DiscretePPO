import torch


class Actor(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        n_actions,
        alpha=3e-4,
        h1_size=256,
        h2_size=256,
        chkpt_dir="weights/actor.pt",
    ):
        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.chkpt_dir = chkpt_dir

        self.h1_layer = torch.nn.Linear(*self.input_dims, self.h1_size)
        self.h2_layer = torch.nn.Linear(self.h1_size, self.h2_size)
        self.output = torch.nn.Linear(self.h2_size, self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), self.alpha, amsgrad=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.h1_layer(x))
        x = torch.nn.functional.tanh(self.h2_layer(x))
        return torch.distributions.Categorical(logits=self.output(x))

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_dir)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_dir))


class Critic(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        alpha=3e-4,
        h1_size=256,
        h2_size=256,
        chkpt_dir="weights/critic.pt",
    ):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.alpha = alpha
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.chkpt_dir = chkpt_dir

        self.h1_layer = torch.nn.Linear(*self.input_dims, self.h1_size)
        self.h2_layer = torch.nn.Linear(self.h1_size, self.h2_size)
        self.output = torch.nn.Linear(self.h2_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), self.alpha, amsgrad=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.h1_layer(x))
        x = torch.nn.functional.tanh(self.h2_layer(x))
        return self.output(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_dir)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_dir))
