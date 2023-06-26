import traceback
import os
import sys
import pathlib

import yaml


BASE_PATH = pathlib.PosixPath(os.path.abspath(__file__)).parent.parent
config_path = BASE_PATH / "config" / "base.yml"

def get_abspath(path):
    if not os.path.isabs(path):
        path = os.path.join(BASE_PATH, path)
    return path


# 载入配置文件的路径
with open(config_path, encoding='utf-8') as f:
    base_conf = yaml.load(f.read(), Loader=yaml.FullLoader)
path_ofa = get_abspath(base_conf['path_ofa'])
path_ckpt = get_abspath(base_conf['path_ckpt'])
path_tokenizer = get_abspath(base_conf['path_tokenizer'])
path_training_config = get_abspath(base_conf['path_training_config'])

# 自动配置ofa
if not os.path.exists(path_ofa):
    os.system(f"cd {BASE_PATH}/attachments && git clone https://github.com/OFA-Sys/OFA.git")

# 自动配置tokenizer
if not os.path.exists(path_tokenizer):
    pass

if path_ofa not in sys.path:
    sys.path.append(path_ofa)

try:
    print("fairseq loading...")
    import ofa_module
    from .module import *
except:
    traceback.print_exc()
