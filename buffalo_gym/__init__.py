from gymnasium.envs.registration import register

register(
    id='Buffalo-v0',
    entry_point='buffalo_gym.envs:BuffaloEnv',
    max_episode_steps=1000
)

register(
    id='MultiBuffalo-v0',
    entry_point='buffalo_gym.envs:MultiBuffaloEnv',
    max_episode_steps=1000
)

register(
    id='BuffaloTrail-v0',
    entry_point='buffalo_gym.envs:BuffaloTrailEnv',
    max_episode_steps=1000
)

register(
    id='Bandit-v0',
    entry_point='buffalo_gym.envs:BuffaloEnv',
    max_episode_steps=1000
)

register(
    id='ContextualBandit-v0',
    entry_point='buffalo_gym.envs:MultiBuffaloEnv',
    max_episode_steps=1000
)

register(
    id='StatefulBandit-v0',
    entry_point='buffalo_gym.envs:BuffaloTrailEnv',
    max_episode_steps=1000
)
