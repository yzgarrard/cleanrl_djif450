from gymnasium.envs.registration import register
from custom_envs.envs.dji_f450 import DJIF450EnvV0p0
from custom_envs.envs.dji_f450_res_acc import DJIF450EnvV1p0
from custom_envs.envs.tacdrone_env import TacDroneHoverEnv
from custom_envs.envs.tacdrone_env_v02 import TacDroneHoverEnvV02
from custom_envs.envs.tacdrone_env_v03 import TacDroneHoverEnvV03

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

register(
    id="custom_envs/TacDroneHover-v0",
    entry_point="custom_envs.envs:TacDroneHoverEnv",
    max_episode_steps=1000,
)

register(
    id="custom_envs/TacDroneHover-v02",
    entry_point="custom_envs.envs:TacDroneHoverEnvV02",
    max_episode_steps=1000,
)

register(
    id="custom_envs/TacDroneHover-v03",
    entry_point="custom_envs.envs:TacDroneHoverEnvV03",
    max_episode_steps=1000,
)