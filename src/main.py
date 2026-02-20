from pathlib import Path
from utils.yaml_parser.configuration import SystemConfiguration

if __name__ == "__main__":
    config_file_path = "../config/default.yaml"
    config_file = Path(config_file_path)
    system_config = SystemConfiguration.parse_config_file(config_file)
    print(system_config)