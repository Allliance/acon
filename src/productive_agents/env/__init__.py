
REGISTERED_ENVS = {}
REGISTERED_ENV_CONFIGS = {}

try:
    from .officebench.env import OfficeBenchEnv
    from .officebench.config  import OfficeBenchEnvConfig
    REGISTERED_ENVS["officebench"] = OfficeBenchEnv
    REGISTERED_ENV_CONFIGS["officebench"] = OfficeBenchEnvConfig
except ImportError:
    print("OfficeBench environment not available, skipping import.")

try:
    from .appworld.env import AppWorldEnv
    from .appworld.config import AppWorldEnvConfig
    REGISTERED_ENVS["appworld"] = AppWorldEnv
    REGISTERED_ENV_CONFIGS["appworld"] = AppWorldEnvConfig
except ImportError:
    print("AppWorld environment not available, skipping import.")

try:
    from .smolagents.env import SmolagentsEnv
    from .smolagents.config import SmolagentsEnvConfig
    REGISTERED_ENVS["smolagents"] = SmolagentsEnv
    REGISTERED_ENV_CONFIGS["smolagents"] = SmolagentsEnvConfig
except ImportError:
    print("Smolagents environment not available, skipping import.")