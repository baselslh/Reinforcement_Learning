## Assignment Description
In this assignment you are supposed to use another environment from Open AI gym classic
environments, which is MountainCar-v0. The target of this game is to climb the hill and
reach the yellow flag.
## Environment Description
You have a car and a mountain hill and your goal is to get an under powered car to the top
of the hill (top = 0.5 position).
- Observations
Type: Box (2)

|Num|Observation|Min|Max|
|:---:|:-------:|:---:|:---:|
|0| position| -1.2| 0.6|
|1 |velocity |-0.07| 0.07|

- Actions
Type: Discrete (3)

|Num|Action|
|:---:|:-------:|
|0 |push left|
|1 |no push|
|2 |push right|

- Reward
Reward is -1 for each time step, until the goal position of 0.5 is reached. There is no
penalty for climbing the left hill, which upon reached acts as a wall.
- Starting State
Random position from -0.6 to -0.4 with no velocity.
- Episode Termination
The episode ends when you reach 0.5 position, or if 200 iterations are reached.
## Assignment Requirements
You are required to deliver the following:
- An implementation in python for Monte Carlo, Q_Learning and SARSA algorithms
based on the MountainCar-v0 environment.
- A comparison between the three algorithms in terms of accuracy and conversion
time (in episodes).