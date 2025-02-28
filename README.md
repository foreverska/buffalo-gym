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

## Check out the [WIKI](https://github.com/foreverska/buffalo-gym/wiki) for fuller documentation.

## Standard Bandit Problems

### Buffalo ("Buffalo-v0" | "Bandit-v0")

Default multi-armed bandit environment.  Arm center values 
are drawn from a normal distribution (0, arms).  When an 
arm is pulled, a random value is drawn from a normal 
distribution (0, 1) and added to the chosen arm center 
value.  This is not intended to be challenging for an agent but 
easy for the debugger to reason about.

### Multi-Buffalo ("MultiBuffalo-v0" | "ContextualBandit-v0")

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

### DuelingBuffalo ("DuelingBuffalo-v0" | "DuelingBandit-v0")

Yue et al. (2012) introduced the dueling bandit variant to model 
situations with only relative feedback.  The agent pulls two levers 
simultaneously; the feedback is whichever lever provides the best 
reward.  This restriction means the agent cannot observe rewards 
and must continually compare arms to determine the best.  Given 
the reward-centric structure of gymnasium returns, we instead 
give a reward of 1 if the first arm chosen was higher than the 
second.  The agent must choose two arms, which cannot be the same.

### BoundlessBuffalo ("BoundlessBuffalo-v0" | "InfiniteArmedBandit-v0")

Built from the Wikipedia entry based on Agrawal, 1995 (Paywalled), 
BoundlessBuffalo approximates the InfiniteArmedBandit problem.  
The reward for this bandit is the action put into a polynomial of 
degree n, with the coefficients randomly sampled from (-0.1, 0.1).  
This environment tests the ability of an algorithm to find an optimal 
input in a continuous space.  The dynamic drawing of new coefficients 
challenges algorithms to adapt to a changing landscape continually.

## Nonstandard Bandit Problems

### Buffalo Trail ("BuffaloTrail-v0" | "StatefulBandit-v0")

A Stateful Bandit builds on the Contextual Bandit by relaxing 
the assumption that rewards depend only on the current state. 
In this framework, the environment incorporates a memory of past 
states, rewarding the maximum to an agent only if it encounters a 
specific sequence of states and selects the correct action.

This setup isolates an agent's ability to track history and infer 
belief states, without introducing the confounding factor of 
exploration, as the agent cannot control state transitions. Stateful 
Bandits provide a targeted environment for studying history-dependent 
decision-making and state estimation.

### Symbolic State ("SymbolicStateBandit-v0")

In real slots, the state of the bandit has little to no impact on 
the underlying rewards.  Plenty of flashing lights and game modes 
serve only to keep the player engaged.  This SymbolicStateBandit 
(SSB) formulation simulates this.  The states do not correlate 
with the underlying rewards in this contextual bandit.

By setting dynamic_rate to None, the rewards are always the same 
despite the changing states; dynamic_rate == pace randomly changes 
the arms with each state, and any other values produce further 
uncorrelated behavior.  This configuration serves as a test bed for 
the "worst case" scenario for a bandit/reinforcement learner.  It 
measures the agent's ability to generalize well and/or how it performs 
when the environment breaks the typical assumptions.

### Tired Buffalo ("TiredBuffalo-v0" | "FatigueBandit-v0")

I asked ChatGPT for a novel bandit formulation.  This bandit is what 
it came up with.  It's hardly novel, though, as it is a special case 
of the "Recovering Bandits" (Pike-Burke & Grünewälder, 2019) problem
where all arms have the same function.  It's on the list of 
non-standard bandits because it's not their problem, but it's hardly 
new.

This bandit problem models resource depletion and recovery. Pulling an arm 
reduces its expected reward ("fatigue"), while unused arms gradually recover. 
Each arm has a unique maximum mean reward, requiring the agent to balance 
immediate rewards against long-term sustainability.

## Using

Install via pip and import buffalo_gym along with gymnasium.

```
import gymnasium  
import buffalo_gym

env = gym.make("Buffalo-v0")
```