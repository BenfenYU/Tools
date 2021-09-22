import os
import shutil

def if_exist_rm_then_make(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)