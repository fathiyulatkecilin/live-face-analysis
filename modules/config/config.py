import os
import yaml
import sys

if len(sys.argv) > 1:
    argv_input_cfg = str(sys.argv[1])
    config_filename = os.path.join("config", "config_" + argv_input_cfg + ".yaml")
    if not os.path.exists(config_filename):
        print("\x1b[0;30;41m" + f" {config_filename} does not exist! " + "\x1b[0m")
        raise -1
else:
    print("\x1b[0;30;41m" + f"Select a config file to load! " + "\x1b[0m")
    raise -1
with open(config_filename, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
