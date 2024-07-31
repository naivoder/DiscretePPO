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
        <td>
            <p><b>BeamRider</b></p>
            <img src="environments/BeamRiderNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Breakout</b></p>
            <img src="environments/BreakoutNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BattleZoneNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BeamRiderNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BreakoutNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table> 
<table>
    <tr>
        <td>
            <p><b>Krull</b></p>
            <img src="environments/KrullNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Berzerk</b></p>
            <img src="environments/BerzerkNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <!--<td>
            <p><b>Breakout</b></p>
            <img src="environments/BreakoutNoFrameskip-v4.gif" width="250" height="250"/>
        </td>-->
    </tr>
    <tr>
        <td>
            <img src="metrics/KrullNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BerzerkNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <!--<td>
            <img src="metrics/BreakoutNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>-->
    </tr>
</table>

---

## Acknowledgements

Special thanks to Phil Tabor, an excellent teacher! I highly recommend his [Youtube channel](https://www.youtube.com/machinelearningwithphil).
