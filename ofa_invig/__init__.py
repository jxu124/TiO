try:
    from ofa_module import *
    from .task_invig import *
    from .dialog_dataset import *
    import ofa_module
    print("invig-fairseq module load success.")

except:
    print("import failed.")
    pass

from ofa_module import *
from .task_invig import *
from .dialog_dataset import *
