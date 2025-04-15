import inspect
import ast
import astunparse
from typing import Dict, Any
import torch
import tvm
from tvm import relax as rx
from tvm.script import relax as R
from tvm.script.ir_builder import relax as relax_builder, ir as I, IRBuilder as IB

def jit(target="cpu"):
    """
    JIT decorator factory function to create a JIT compilation decorator
    
    Args:
        target (str): Target device, supports 'cpu' or 'gpu'
    
    Returns:
        function: A decorator function for JIT compilation
    """
    assert target in ["cpu", "gpu"]
    def inner(fn):
        return JIT(fn, target=target)
    return inner

class JIT:
    """
    JIT compilation class that compiles Python functions to TVM Relax IR
    
    Attributes:
        fn (function): Python function to be compiled
        target (str): Target device type
    """
    def __init__(self, fn, target="cpu"):
        self.fn = fn
        self.target = target
    
    def __call__(self, *args, **kwargs):
        """
        Compile and execute the Python function
        
        Args:
            *args: Function arguments
            **kwargs: Keyword arguments
            
        Returns:
            Any: Execution result of the compiled function
        """
        # Get function source code and parse to AST
        fn_src = inspect.getsource(self.fn)
        fn_ast = ast.parse(fn_src)
        print(ast.dump(fn_ast))
        
        # Get function global context
        ctx = self.fn.__globals__.copy()
        
        # Generate code and compile
        code_generator = CodeGenerator(fn_ast, ctx, self.target)
        compiled_kernel = code_generator.code_gen()
        
        # Prepare input arguments
        input_args = []
        for arg in args:
            input_args.append(arg.data)
            
        # Execute compiled function
        return compiled_kernel(*input_args)

class CodeGenerator(ast.NodeVisitor):
    """
    AST visitor class that converts Python AST to TVM Relax IR
    
    Attributes:
        fn_ast (ast.AST): Function AST
        ctx (dict): Function context
        target (str): Target device
        ib (IRBuilder): TVM IR builder
        ir_module: Relax IR module
        entry (str): Entry function name
        ret: Return value
        local_var_table (Dict): Local variable table
    """
    def __init__(self, fn_ast, ctx, target):
        self.fn_ast = fn_ast
        self.ctx = ctx
        self.target = target
        self.ib = IB()
        self.ir_module = None
        self.entry = None
        self.ret = None
        self.local_var_table : Dict[str, Any] = {}
    
    def code_gen(self):
        """
        Generate Relax IR code
        
        Returns:
            function: Compiled function
        """
        # Generate Relax IR module
        with self.ib:
            self.visit(self.fn_ast)
        module = self.ib.get()
        print(module)

        # Apply optimization passes
        with tvm.transform.PassContext(opt_level=3):
            # Apply Opt Pass
            seq = tvm.transform.Sequential(
                [
                    rx.transform.LegalizeOps(),
                ])
            module = seq(module)
        print("After applied passes...")    
        print(module)

        # Compile and create virtual machine
        mapped_target = {'cpu': 'llvm', 'gpu': 'cuda'}
        target = tvm.target.Target(mapped_target[self.target])
        with tvm.transform.PassContext(opt_level=3):
            ex = rx.build(module, target=target)
        device = tvm.cuda() if "cuda" in target.keys else tvm.cpu()
        vm = rx.VirtualMachine(ex, device=device)
        return vm[self.entry]

    def visit(self, node):
        """
        Visit AST node
        
        Args:
            node (ast.AST): AST node
            
        Returns:
            Any: Visit result
        """
        print("Visit " + node.__class__.__name__)
        return super().visit(node)
    
    def visit_Module(self, node: ast.Module):
        """
        Visit module node
        
        Args:
            node (ast.Module): Module node
        """
        if self.ir_module:
            raise AssertionError("We should have only one module!")
        self.ir_module = I.ir_module()
        with self.ir_module:
            super().generic_visit(node)
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visit function definition node
        
        Args:
            node (ast.FunctionDef): Function definition node
        """
        fn = relax_builder.function()
        self.entry = node.name
        with fn:
            R.func_name(node.name)
            self.visit(node.args)
            self._visit_compound_stmt(node.body)

            if self.ret is None:
                R.func_ret_value(rx.ShapeExpr([]))
            else:
                R.func_ret_value(self.ret)
    
    def visit_arguments(self, node: ast.arguments):
        """
        Visit function arguments node
        
        Args:
            node (ast.arguments): Arguments node
        """
        for arg in node.args:
            if arg.annotation is None:
                raise ValueError(arg, "Type annotation is required for function parameters.")
            arg_name = arg.arg
            anno = eval(astunparse.unparse(arg.annotation), self.ctx)
            param = R.arg(arg_name, R.Tensor(shape=anno.shape, dtype=anno.dtype))
            self.local_var_table[arg_name] = param

    def visit_Pass(self, node: ast.Pass):
        """Visit pass node"""
        pass

    def visit_Assign(self, node: ast.Assign):
        """
        Visit assignment node
        
        Args:
            node (ast.Assign): Assignment node
        """
        if len(node.targets) != 1:
            raise NotImplementedError("Doesn't support simultaneous multiple assignment like 'a = b = c' in AST node type: {}".format(type(node).__name__))
        target: rx.Var = self.visit(node.targets[0])
        value = self.visit(node.value)
        self.local_var_table[target.name_hint] = value
        self.ib.name(target.name_hint, value)
    
    def visit_Name(self, node: ast.Name):
        """
        Visit name node
        
        Args:
            node (ast.Name): Name node
            
        Returns:
            rx.Var: Corresponding Relax variable
        """
        name = node.id
        if isinstance(node.ctx, ast.Store):
            if name not in self.local_var_table.keys():
                self.local_var_table[name] = rx.Var(name, struct_info=rx.ObjectStructInfo())
        return self.local_var_table[name]
    
    def visit_BinOp(self, node: ast.BinOp):
        """
        Visit binary operation node
        
        Args:
            node (ast.BinOp): Binary operation node
            
        Returns:
            Any: Operation result
        """
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        return R.emit(self._binOp_maker(node.op)(lhs, rhs))
    
    def visit_Return(self, node: ast.Return):
        """
        Visit return node
        
        Args:
            node (ast.Return): Return node
            
        Returns:
            Any: Return value
        """
        ret_value = self.visit(node.value)
        return ret_value
    
    def visit_Constant(self, node: ast.Constant):
        """
        Visit constant node
        
        Args:
            node (ast.Constant): Constant node
            
        Returns:
            Any: Constant value
        """
        return R.emit(rx.const(node.value))
        
    def _visit_compound_stmt(self, stmts):
        """
        Visit compound statement
        
        Args:
            stmts (list): Statement list
        """
        assert isinstance(stmts, (list, tuple))
        for stmt in stmts:
            ret = self.visit(stmt)
            if ret is not None and isinstance(stmt, ast.Return):
                self.ret = ret
    
    def _binOp_maker(self, node: ast.operator):
        """
        Create binary operation
        
        Args:
            node (ast.operator): Operator node
            
        Returns:
            function: Corresponding Relax operation function
        """
        if isinstance(node, ast.Add):
            return R.add
        else:
            raise NotImplementedError("Unsupported AST node type: {}".format(type(node).__name__))
    
    def generic_visit(self, node: ast.AST):
        """
        Generic visit method
        
        Args:
            node (ast.AST): AST node
        """
        raise NotImplementedError("Unsupported AST node type: {}".format(type(node).__name__))

class Tensor:
    """
    Tensor class for representing tensor data
    
    Attributes:
        shape (tuple): Tensor shape
        dtype (str): Data type
        _data: Actual data
    """
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self._data = None
    
    @property
    def data(self):
        """
        Get tensor data
        
        Returns:
            Any: Tensor data
        """
        return self._data
    
    @data.setter
    def data(self, data: "torch.Tensor"):
        """
        Set tensor data, convert PyTorch tensor to TVM tensor
        
        Args:
            data (torch.Tensor): PyTorch tensor
        """
        def _from_dlpack(tensor):
            """
            Convert PyTorch tensor to TVM tensor
            
            Args:
                tensor (torch.Tensor): PyTorch tensor
                
            Returns:
                tvm.nd.NDArray: TVM tensor
            """
            from tvm.runtime import Device
            from tvm.runtime import ndarray
            try:
                return ndarray.from_dlpack(tensor)
            except RuntimeError:
                pass
            device_type = tensor.device.type
            device_id = tensor.device.index or 0
            return ndarray.array(
                tensor.numpy(),
                device=Device(
                    Device.STR2MASK[device_type],
                    device_id,
                ),
            )
        data = _from_dlpack(data)
        if data.shape != tuple(self.shape):
            raise ValueError(f"Shape mismatch: expected {tuple(self.shape)}, got {data.shape}")
        if data.dtype != self.dtype:
            raise ValueError(f"Dtype mismatch: expected {self.dtype}, got {data.dtype}")
        self._data = data

    def __str__(self):
        """
        Return string representation of tensor
        
        Returns:
            str: String representation of tensor
        """
        return str(self.dtype) + '[' + ', '.join(str(s) for s in self.shape) + ']'

# Test code
@jit(target="cpu")
def add(a: Tensor(shape=(2, 3), dtype="float32"), b: Tensor(shape=(2, 3), dtype="float32")):
    """
    Test function: add two tensors
    
    Args:
        a (Tensor): First tensor
        b (Tensor): Second tensor
        
    Returns:
        Tensor: Sum of tensors
    """
    out = a + b
    return out

# Create test tensors
a = Tensor(shape=(2, 3), dtype="float32")
b = Tensor(shape=(2, 3), dtype="float32")
a.data = torch.ones(size=(2, 3), dtype=torch.float32)
b.data = torch.ones(size=(2, 3), dtype=torch.float32)
print(add(a, b))