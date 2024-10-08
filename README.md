# Proximal Policy Optimization (Discrete)

## Overview

🚧 🛠️👷‍♀️ 🛑 Under construction...

This repository contains an implementation of Proximal Policy Optimization (PPO) for discrete action spaces, which has been evaluated against a variety of Gymnasium and Atari environments.  

The main script in its current form is configured for Atari environments, with a custom environment wrapper that follows the approach outlined in the original DQN paper (for this reason, it is recommended to use the 'NoFrameskip' versions of the environments).  

## Setup

### Required Dependencies

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Running the Algorithm

You can run the algorithm on any supported Gymnasium environment. For example:

```bash
python main.py --env 'MsPacmanNoFrameskip-v4'
```

## Results
#### 🤔 For your consideration: 
The Atari environments were trained for 20000 games. I regret this decision as it lead to inconsistent numbers of learning steps between environments (due to some games requiring more/less steps per game).  

I also did not use reward scaling, which I use for most other algorithms. This was a nearly arbitrary decision that came about due to initial debugging - at a certain point things suddenly began to work so I just kinda rolled with it...

I only started tracking the average critic value for a set of fixed states after many environments had already been trained, but I feel that this provides an additional interesting piece of context. 

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
            <img src="metrics/BreakoutNoFrameskip-v4_metrics.png" width="250" height="250"/>
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
        <td>
            <p><b>CrazyClimber</b></p>
            <img src="environments/CrazyClimberNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/KrullNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BerzerkNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/CrazyClimberNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>DemonAttack</b></p>
            <img src="environments/DemonAttackNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Kangaroo</b></p>
            <img src="environments/KangarooNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>KungFuMaster</b></p>
            <img src="environments/KungFuMasterNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/DemonAttackNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/KangarooNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/KungFuMasterNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Zaxxon</b></p>
            <img src="environments/ZaxxonNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Skiing</b></p>
            <img src="environments/SkiingNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>MontezumaRevenge</b></p>
            <img src="environments/MontezumaRevengeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/ZaxxonNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/SkiingNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/MontezumaRevengeNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Bowling</b></p>
            <img src="environments/BowlingNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Boxing</b></p>
            <img src="environments/BoxingNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Carnival</b></p>
            <img src="environments/CarnivalNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BowlingNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BoxingNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/CarnivalNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Centipede</b></p>
            <img src="environments/CentipedeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>ChopperCommand</b></p>
            <img src="environments/ChopperCommandNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Defender</b></p>
            <img src="environments/DefenderNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/CentipedeNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/ChopperCommandNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/DefenderNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>DoubleDunk</b></p>
            <img src="environments/DoubleDunkNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>NameThisGame</b></p>
            <img src="environments/NameThisGameNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Solaris</b></p>
            <img src="environments/SolarisNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/DoubleDunkNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/NameThisGameNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/SolarisNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>SpaceInvaders</b></p>
            <img src="environments/SpaceInvadersNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Phoenix</b></p>
            <img src="environments/PhoenixNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>StarGunner</b></p>
            <img src="environments/StarGunnerNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/SpaceInvadersNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/PhoenixNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/StarGunnerNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Pitfall</b></p>
            <img src="environments/PitfallNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Tennis</b></p>
            <img src="environments/TennisNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Pong</b></p>
            <img src="environments/PongNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/PitfallNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/TennisNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/PongNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Pooyan</b></p>
            <img src="environments/PooyanNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>TimePilot</b></p>
            <img src="environments/TimePilotNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Tutankham</b></p>
            <img src="environments/TutankhamNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/PooyanNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/TimePilotNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/TutankhamNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Enduro</b></p>
            <img src="environments/EnduroNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>UpNDown</b></p>
            <img src="environments/UpNDownNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>PrivateEye</b></p>
            <img src="environments/PrivateEyeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/EnduroNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/UpNDownNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/PrivateEyeNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Qbert</b></p>
            <img src="environments/QbertNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Riverraid</b></p>
            <img src="environments/RiverraidNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>RoadRunner</b></p>
            <img src="environments/RoadRunnerNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/QbertNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/RiverraidNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/RoadRunnerNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>FishingDerby</b></p>
            <img src="environments/FishingDerbyNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Venture</b></p>
            <img src="environments/VentureNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Freeway</b></p>
            <img src="environments/FreewayNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/FishingDerbyNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/VentureNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/FreewayNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Seaquest</b></p>
            <img src="environments/SeaquestNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Robotank</b></p>
            <img src="environments/RobotankNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Frostbite</b></p>
            <img src="environments/FrostbiteNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/SeaquestNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/RobotankNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/FrostbiteNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>VideoPinball</b></p>
            <img src="environments/VideoPinballNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Gopher</b></p>
            <img src="environments/GopherNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Gravitar</b></p>
            <img src="environments/GravitarNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/VideoPinballNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/GopherNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/GravitarNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>WizardOfWor</b></p>
            <img src="environments/WizardOfWorNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Hero</b></p>
            <img src="environments/HeroNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>YarsRevenge</b></p>
            <img src="environments/YarsRevengeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/WizardOfWorNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/HeroNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/YarsRevengeNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>ElevatorAction</b></p>
            <img src="environments/ElevatorActionNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>IceHockey</b></p>
            <img src="environments/IceHockeyNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Jamesbond</b></p>
            <img src="environments/JamesbondNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/ElevatorActionNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/IceHockeyNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/JamesbondNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>JourneyEscape</b></p>
            <img src="environments/JourneyEscapeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/JourneyEscapeNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>

## Acknowledgements

Special thanks to Phil Tabor, an excellent teacher! I highly recommend his [Youtube channel](https://www.youtube.com/machinelearningwithphil).
