import inspect
import ast
import astunparse
from typing import Dict, Any
import torch
import tvm
from tvm import dlight as dl
from tvm import relax as rx
from tvm.script import relax as R
from tvm.script.ir_builder import relax as relax_builder, ir as I, IRBuilder as IB
def jit(target="cpu"):
    assert target in ["cpu", "gpu"]
    def inner(fn):
        return JIT(fn, target=target)
    return inner

class JIT:
    def __init__(self, fn, target="cpu"):
        self.fn = fn
        self.target = target
    
    def __call__(self, *args, **kwargs):
        fn_src = inspect.getsource(self.fn)
        fn_ast = ast.parse(fn_src)
        print(ast.dump(fn_ast))
        ctx = self.fn.__globals__.copy()
        code_generator = CodeGenerator(fn_ast, ctx, self.target)
        compiled_kernel = code_generator.code_gen()
        input_args = []
        for arg in args:
            input_args.append(arg.data)
        return compiled_kernel(*input_args)

class CodeGenerator(ast.NodeVisitor):
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
        with self.ib:
            self.visit(self.fn_ast)
        module = self.ib.get()
        print(module)

        # apply transform pass on module
        with tvm.transform.PassContext(opt_level=3):
            # Apply Opt Pass
            seq = tvm.transform.Sequential(
                [
                    rx.transform.ConvertToDataflow(),
                    rx.transform.LegalizeOps(),
                    rx.transform.AnnotateTIROpPattern(),
                    rx.transform.FuseOps(),
                    rx.transform.FuseTIR(),
                ])
            module = seq(module)
        print("After applied passes...")    
        print(module)

        mapped_target = {'cpu': 'llvm', 'gpu': 'cuda'}
        target = tvm.target.Target(mapped_target[self.target])

        # use delight to auto schedule
        if "cuda" in target.keys:
            with target:
                module = dl.ApplyDefaultSchedule(dl.gpu.Fallback(),)(module)
            print("After applied dlight...")
            print(module)
        
        with tvm.transform.PassContext(opt_level=3):
            ex = rx.build(module, target=target)
        
        if "cuda" in target.keys:
            # dump cuda source
            print(ex.mod.imported_modules[0].imported_modules[0].get_source())

        device = tvm.cuda() if "cuda" in target.keys else tvm.cpu()
        vm = rx.VirtualMachine(ex, device=device)
        return vm[self.entry]


    def visit(self, node):
        print("Visit " + node.__class__.__name__)
        return super().visit(node)
    
    def visit_Module(self, node: ast.Module):
        if self.ir_module:
            raise AssertionError("We should have only one module!")
        self.ir_module = I.ir_module()
        with self.ir_module:
            super().generic_visit(node)
        
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
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
        for arg in node.args:
            if arg.annotation is None:
                raise ValueError(arg, "Type annotation is required for function parameters.")
            arg_name = arg.arg
            anno = eval(astunparse.unparse(arg.annotation), self.ctx)
            param = R.arg(arg_name, R.Tensor(shape=anno.shape, dtype=anno.dtype))
            self.local_var_table[arg_name] = param

    def visit_Pass(self, node: ast.Pass):
        pass

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            raise NotImplementedError("Doesn't support simultaneous multiple assignment like 'a = b = c' in AST node type: {}".format(type(node).__name__))
        target: rx.Var = self.visit(node.targets[0])
        value = self.visit(node.value)
        self.local_var_table[target.name_hint] = value
        self.ib.name(target.name_hint, value)
    
    def visit_Name(self, node: ast.Name):
        name = node.id
        if isinstance(node.ctx, ast.Store):
            if name not in self.local_var_table.keys():
                self.local_var_table[name] = rx.Var(name, struct_info=rx.ObjectStructInfo())
        return self.local_var_table[name]
    
    def visit_BinOp(self, node: ast.BinOp):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        return R.emit(self._binOp_maker(node.op)(lhs, rhs))
    
    def visit_Return(self, node: ast.Return):
        ret_value = self.visit(node.value)
        return ret_value
    
    def visit_Constant(self, node: ast.Constant):
        return R.emit(rx.const(node.value))
        
    def _visit_compound_stmt(self, stmts):
        assert isinstance(stmts, (list, tuple))
        for stmt in stmts:
            ret = self.visit(stmt)
            if ret is not None and isinstance(stmt, ast.Return):
                self.ret = ret
    
    def _binOp_maker(self, node: ast.operator):
        if isinstance(node, ast.Add):
            return R.add
        else:
            raise NotImplementedError("Unsupported AST node type: {}".format(type(node).__name__))
    
    def generic_visit(self, node: ast.AST):
        raise NotImplementedError("Unsupported AST node type: {}".format(type(node).__name__))

class Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self._data = None
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data: "torch.Tensor"):
        def _from_dlpack(tensor):
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
        return str(self.dtype) + '[' + ', '.join(str(s) for s in self.shape) + ']'

@jit(target="gpu")
def add(a: Tensor(shape=(2, 3), dtype="float32"), b: Tensor(shape=(2, 3), dtype="float32")):
    out = a + b
    out = out + a
    return out

a = Tensor(shape=(2, 3), dtype="float32")
b = Tensor(shape=(2, 3), dtype="float32")
a.data = torch.ones(size=(2, 3), dtype=torch.float32, device="cuda")
b.data = torch.ones(size=(2, 3), dtype=torch.float32, device="cuda")
print(add(a, b))