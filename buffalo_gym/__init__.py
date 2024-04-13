from gymnasium.envs.registration import register

register(
    id='Buffalo-v0',
    entry_point='buffalo_gym.envs:BuffaloEnv',
    max_episode_steps=1000
)
