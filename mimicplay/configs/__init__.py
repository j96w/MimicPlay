from mimicplay.configs.config import Config
from mimicplay.configs.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from mimicplay.configs.mimicplay_config import MimicPlayConfig