
from .utils import *
from x2paddle.utils import *


class FuncDemo(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        """ 更新参数。
        """
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "other", "y")

    def delete_attrs(self):
        """ 删除参数。
        """
        delete_key(self.kwargs, "out")

    def check_attrs(self):
        """ 确认参数的值。
        """
        pass

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.matmul"):
            # 作用：当出现可变参数或关键字参数，无法对参数进行处理；
            # 需要根据x2paddle封装的对应API命名生成代码(x2paddle封装的对应API相关代码在步骤3中实现)
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # 作用：将paddle与pytorch不同的可变参数替换成字典参数，并生成相应代码
            self.convert_args2kwargs()
            return self.convert_to_paddle()

class FuncEmpty(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
    def delete_attrs(self):
                """ 删除参数。
                """
             
                delete_key(self.kwargs, "out")
                delete_key(self.kwargs, "dtype")
                delete_key(self.kwargs, "layout")
                delete_key(self.kwargs, "device")
                delete_key(self.kwargs, "requires_grad")
                delete_key(self.kwargs, "pin_memory")

    def process_attrs(self):
        rename_key(self.kwargs, "size", "shape")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.empty"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs()
            return self.convert_to_paddle()


class FuncRandint(Mapper):
    """ torch.randint
    """
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
    def delete_attrs(self):
            """ 删除参数。
            """
            delete_key(self.kwargs, "generator")
            delete_key(self.kwargs, "out")
            delete_key(self.kwargs, "dtype")
            delete_key(self.kwargs, "layout")
            delete_key(self.kwargs, "device")
            delete_key(self.kwargs, "requires_grad")

    def process_attrs(self):
        rename_key(self.kwargs, "size", "shape")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.randint"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs()
            return self.convert_to_paddle()


class FuncChunk(Mapper):
    """ torch.randint
    """
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
   
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "dim", "axis")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.chunk"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs()
            return self.convert_to_paddle()


class FuncUnfold(Mapper):
    """ torch.randint
    """
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
   
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "padding", "paddings")
        rename_key(self.kwargs, "stride", "strides")
        rename_key(self.kwargs, "dilation", "dilations")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.unfold"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs()
            return self.convert_to_paddle()