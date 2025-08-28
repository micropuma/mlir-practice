#ifndef DLY_DIALECT_TD
#define DLY_DIALECT_TD

include "mlir/IR/OpBase.td"

def DlyDialect : Dialect {
  let name = "dly";
  let cppNamespace = "::mlir::toy";
  let summary = "A high-level dialect for analyzing and optimizing the dly language";
  // A much longer description of our dialect.
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }]; 
}

#endif // DlyDialect_TD
