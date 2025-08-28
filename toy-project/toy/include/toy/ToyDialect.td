#ifndef TOY_DIALECT_TD
#define TOY_DIALECT_TD

include "mlir/IR/OpBase.td"

def ToyDialect : Dialect {
  let name = "toy";
  let cppNamespace = "::toy";
  let summary = "A high-level dialect for analyzing and optimizing the Toy language";
  // A much longer description of our dialect.
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }];
  let useDefaultTypePrinterParser = true;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // ToyDialect_TD
