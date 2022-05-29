
from .utils import *
from x2paddle.utils import *


class FuncEmpty(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "device", "place")

    def run(self):
        if self.rename_func_name("paddle.to_tensor"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs()
            return self.convert_to_paddle()