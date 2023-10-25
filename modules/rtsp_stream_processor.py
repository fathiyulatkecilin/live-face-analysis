import os
import sys
import time
import subprocess

try:
	python_path = (subprocess.check_output("which python", shell=True).strip()).decode('utf-8')
	while True:
		print("Run", sys.argv[1:])
		argvs = []
		for arg in sys.argv[1:]:
			if 'rtsp' in arg:
				new = F'\'{arg}\''
				argvs.append(new)
			else:
				argvs.append(arg)

		command = F"{python_path} {' '.join(argvs)}"
		print(command)
		os.system(command)
		time.sleep(5)

except KeyboardInterrupt:
	print("Done")