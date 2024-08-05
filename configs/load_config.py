import yaml

# read config file
with open("configs/case_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

CONFIG["data_path"] = CONFIG["data_path_template"].format(case_name=CONFIG["case_name"])
CONFIG["temp_file_path"] = CONFIG["temp_file_path_template"].format(
    case_name=CONFIG["case_name"]
)
