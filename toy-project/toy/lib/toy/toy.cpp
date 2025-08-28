#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "toy/ToyDialect.h"
#include "toy/ToyOps.h"

#include "toy/ToyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "toy/Toy.cpp.inc"

#include "toy/ToyTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#define GET_TYPEDEF_CLASSES
#include "toy/ToyTypes.cpp.inc"

using namespace mlir;
using namespace toy;

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Toy.cpp.inc"
  >();
  registerTypes();
}

void ToyDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "toy/ToyTypes.cpp.inc"
      >();
}

// 实现SubOp的verifier
LogicalResult SubOp::verify() {
  if (getLhs().getType() != getRhs().getType())
    return this->emitError() << "Lhs type " << getLhs().getType() << " and Rhs type " << getRhs().getType() << " must be the same";
  return success();
}

// 实现ConstantOp的返回值推断
mlir::LogicalResult ConstantOp::inferReturnTypes(
  mlir::MLIRContext * context,
  std::optional<mlir::Location> location,
  Adaptor adaptor,
  llvm::SmallVectorImpl<mlir::Type> & inferedReturnType) {
    auto type = adaptor.getValueAttr().getType();
    inferedReturnType.push_back(type);
    return mlir::success();
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](auto & builder, auto argTypes, auto results, auto, auto) {
    return builder.getFunctionType(argTypes, results);
  };
  return function_interface_impl::parseFunctionOp(
    parser, result, false, 
    getFunctionTypeAttrName(result.name), buildFuncType, 
    getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name)
  );
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
    p, *this, false, getFunctionTypeAttrName(),
    getArgAttrsAttrName(), getResAttrsAttrName());
}