import os
import sys
import json
import subprocess


if len(sys.argv)<3:
    print("Run with this command: python pm2_run.py <name-service> <name-file_script.py> [argv]")
    sys.exit()

python_path = subprocess.check_output("which python", shell=True).strip()
python_path = python_path.decode("utf-8")

pwd = subprocess.check_output("pwd", shell=True).strip()
pwd = pwd.decode("utf-8")
script = os.path.join(pwd, "modules", "rtsp_stream_processor.py")

pm2_configs = {
	"name"          : sys.argv[1],
	"autorestart"	: True,
    "interpreter"   : python_path,
    "script"        : script,
    "args"          : " ".join(sys.argv[2:]),
    "cwd"           : pwd,
}

pm2_configs = [pm2_configs]
with open("run_services.json", 'w', encoding="utf-8") as f:
    json.dump(pm2_configs, f, indent=4)

os.system(F"pm2 start run_services.json")

print("____________________")
print(pm2_configs)
print("____________________")