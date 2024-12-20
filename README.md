# Buffalo Gym

A multi-armed bandit (MAB) environment for the gymnasium API.
One-armed Bandit is a reference to slot machines, and Buffalo 
is a reference to one such slot machine that I am fond 
of.  MABs are an excellent playground for theoretical exercise and 
debugging of RL agents as they provide an environment that 
can be reasoned about easily.  It helped me once to step back 
and write an MAB to debug my DQN agent.  But there was a lack 
of native gymnasium environments, so I wrote Buffalo, an easy-to-use 
 environment that it might help someone else.

## Buffalo ("Buffalo-v0" | "Bandit-v0")

Default multi-armed bandit environment.  Arm center values 
are drawn from a normal distribution (0, arms).  When an 
arm is pulled, a random value is drawn from a normal 
distribution (0, 1) and added to the chosen arm center 
value.  This is not intended to be challenging for an agent but 
easy for the debugger to reason about.

## Multi-Buffalo ("MultiBuffalo-v0" | "ContextualBandit-v0")

This serves as a contextual bandit implementation.  It is a 
k-armed bandit with n states.  These states are indicated to 
the agent in the observation and the two states have different 
reward offsets for each arm.  The goal of the agent is to 
learn and contextualize best action for a given state.  This is 
a good stepping stone to Markov Decision Processes.

This module had an extra parameter, pace.  By default (None), a 
new state is chosen for every step of the environment.  It can 
be set to any integer to determine how many steps between randomly 
choosing a new state.  Of course, transitioning to a new state is 
not guaranteed as the next state is random.

## Buffalo Trail ("BuffaloTrail-v0" | "StatefulBandit-v0")

This serves as a stateful bandit implementation.  There is a 
pervasive rumor that slot machine manufacturers put in 
a secret sequence of bets which trigger a large reward or the 
jackpot.  It is almost certainly not true in the real world but 
it is here.  A sequence of actions gives the max reward.  The 
sequence is randomly chosen on environment setup and indicated 
in the info of reset.  Not all sequences are aliased and this 
may be an important thing to check in an implementation.  Therefore, 
there is a rudimentary algorithm to force aliasing included.

## Using

Install via pip and import buffalo_gym along with gymnasium.

```
import gymnasium  
import buffalo_gym

env = gym.make("Buffalo-v0")
```