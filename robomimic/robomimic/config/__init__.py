from robomimic.config.config import Config
from robomimic.config.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from robomimic.config.bc_config import BCConfig
from robomimic.config.bcq_config import BCQConfig
from robomimic.config.cql_config import CQLConfig
from robomimic.config.iql_config import IQLConfig
from robomimic.config.gl_config import GLConfig
from robomimic.config.hbc_config import HBCConfig
from robomimic.config.iris_config import IRISConfig
from robomimic.config.td3_bc_config import TD3_BCConfig

from robomimic.config.mini_bet_config import MiniBETConfig
from robomimic.config.mini_bet_vision_config import MiniBETVisionConfig
from robomimic.config.mini_bet_hybrid_config import MiniBETHybridConfig
from robomimic.config.ibc_config import IBCConfig
from robomimic.config.dp_state_config import DPStateConfig
from robomimic.config.dp_vision_config import DPVisionConfig
from robomimic.config.dp_hybrid_config import DPHybridConfig
