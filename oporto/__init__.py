from gymnasium.envs.registration import register

register(
    id="BotvinickFourRooms-v0",
    entry_point="oporto.envs:FourRoomsEnv",
    max_episode_steps=600,
)
