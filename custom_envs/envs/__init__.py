from gymnasium.envs.registration import register
from custom_envs.envs.dji_f450 import DJIF450EnvV0p0
from custom_envs.envs.dji_f450_res_acc import DJIF450EnvV1p0

register(
    id="custom_envs/DJIF450-v0",
    entry_point="custom_envs.envs:DJIF450EnvV0p0",
    max_episode_steps=1000,
)

register(
    id="custom_envs/DJIF450-v1",
    entry_point="custom_envs.envs:DJIF450EnvV1p0",
    max_episode_steps=1000,
)