# su
# /data/yujingbo/miniconda3/bin/python clear_vanish_cache.py 0
import subprocess
import re
import argparse

from tqdm import tqdm

paser = argparse.ArgumentParser()
paser.add_argument("which_gpu")
args = paser.parse_args()

number = args.which_gpu
return_value = subprocess.check_output(f"fuser -v /dev/nvidia{number}",shell=True,stderr=subprocess.STDOUT)

pattern = re.compile(r'[a-z]+\s+([0-9]+)') 
re_results = pattern.findall(str(return_value))
for re_result in tqdm(re_results):
    subprocess.run(["kill", "-9", f"{re_result}"])

