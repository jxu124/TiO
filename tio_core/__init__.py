import traceback
import os
import sys
import pathlib

import yaml


def check_path(path):
    if not os.path.exists(path):
        path_abs = os.path.join(BASE_PATH, path)
        if not os.path.exists(path_abs):
            raise FileNotFoundError(f"{path} not found.")
        return path_abs
    return path


BASE_PATH = pathlib.PosixPath(os.path.abspath(__file__)).parent.parent
config_path = BASE_PATH / "config" / "base.yml"

with open(config_path, encoding='utf-8') as f:
    base_conf = yaml.load(f.read(), Loader=yaml.FullLoader)
path_ofa = check_path(base_conf['path_ofa'])
path_ckpt = check_path(base_conf['path_ckpt'])
path_tokenizer = check_path(base_conf['path_tokenizer'])
path_config = check_path(base_conf['path_config'])

if path_ofa not in sys.path:
    sys.path.append(path_ofa)

try:
    print("fairseq loading...")
    import ofa_module
    from .module import *
except:
    traceback.print_exc()
