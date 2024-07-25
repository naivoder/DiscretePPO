# Proximal Policy Optimization (Discrete)

## Overview

🚧 🛠️👷‍♀️ 🛑 Under construction...

Poor learning on lots of environments even when using the NoFrameskip envs...  
Reading lots of commentary on how the authors used a bunch of tricks they didn't include in the paper. I might have to do some fancier stuff to ensure the models learn effectively 🧐

## Setup

### Required Dependencies

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Running the Algorithm

You can run the algorithm on any supported Gymnasium environment. For example:

```bash
python main.py --env 'LunarLander-v2'
```

---

<table>
    <tr>
        <td>
            <p><b>CartPole-v1</b></p>
            <img src="environments/CartPole-v1.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>MountainCar-v0</b></p>
            <img src="environments/MountainCar-v0.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Acrobot-v1</b></p>
            <img src="environments/Acrobot-v1.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/CartPole-v1_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/MountainCar-v0_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Acrobot-v1_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>LunarLander-v2</b></p>
            <img src="environments/LunarLander-v2.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>AirRaid</b></p>
            <img src="environments/AirRaidNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Alien</b></p>
            <img src="environments/AlienNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/LunarLander-v2_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AirRaidNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AlienNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Amidar</b></p>
            <img src="environments/AmidarNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Assault</b></p>
            <img src="environments/AssaultNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Asterix</b></p>
            <img src="environments/AsterixNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/AmidarNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AssaultNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AsterixNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Asteroids</b></p>
            <img src="environments/AsteroidsNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Atlantis</b></p>
            <img src="environments/AtlantisNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>BankHeist</b></p>
            <img src="environments/BankHeistNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
    <td>
            <img src="metrics/AsteroidsNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AtlantisNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BankHeistNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>BattleZone</b></p>
            <img src="environments/BattleZoneNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <!-- <td>
            <p><b>SpaceInvaders-v5</b></p>
            <img src="environments/SpaceInvaders-v5.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Tetris-v5</b></p>
            <img src="environments/Tetris-v5.gif" width="250" height="250"/>
        </td> -->
    </tr>
    <tr>
        <td>
            <img src="metrics/BattleZoneNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <!-- <td>
            <img src="metrics/SpaceInvaders-v5_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Tetris-v5_running_avg.png" width="250" height="250"/>
        </td> -->
    </tr>
</table> 

It's very interesting that PPO struggles to solve the MountainCar environment (solved easily by DDPG). I found this comment from `/u/jurniss` on Reddit very insightful:

> Sparse rewards. In OpenAI Gym MountainCar you only get a positive reward when  you reach the top. PPO is an on-policy algorithm. It performs a policy gradient update after each episode and throws the data away. Reaching the goal in MountainCar by random actions is a pretty rare event. When it finally happens, it's very unlikely that a single policy gradient update will be enough to start reaching the goal consistently, so PPO gets stuck again with no learning signal until it reaches the goal again by chance. On the other hand, DDPG stores this event in the replay buffer so it does not forget. The TD bootstrapping of the Q function will eventually propagate the reward from the goal backwards into the Q estimate for other states near the goal This is a big advantage of off-policy RL algorithms. Also DDPG uses an Ornstein-Uhlenbeck process for time-correlated exploration, whereas PPO samples Gaussian noise. The Ornstein-Uhlenbeck process is more likely to generate useful exploratory actions. (The exploration methods are not immutable properties of the algorithms, just the Baselines implementations.)

---

## Acknowledgements

Special thanks to Phil Tabor, an excellent teacher! I highly recommend his [Youtube channel](https://www.youtube.com/machinelearningwithphil).
