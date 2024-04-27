from gymnasium.envs.registration import register

register(
    id="BotvinickFourRooms-v0",
    entry_point="optimal_portable_options.envs:FourRoomsEnv",
    max_episode_steps=600,
)
