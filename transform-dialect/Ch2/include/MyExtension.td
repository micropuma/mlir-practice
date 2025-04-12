//===-- MyExtension.td - Transform dialect tutorial --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the
// Chapter 2 of the Transform dialect tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef MY_EXTENSION
#define MY_EXTENSION

include "MyExtensionTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// chapter3针对这个operation做了一些修改
// 主要通过引入TransformEachOpTrait的方式减少skeleton 重复code的编写
// Define the new operation. By convention, prefix its name with the name of the dialect 
// extension, "my.". The full operation name will be further prefixed with "transform.".
def ChangeCallTargetOp : Op<Transform_Dialect, "my.change_call_target",
    // Indicate that the operation implements the required TransformOpInterface and
    // MemoryEffectsOpInterface.
    // 这里做修改，增添transform each op trait。会自动对每一个call处理，不用手动遍历
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  // Provide a brief and a full description. It is recommended that the latter describes 
  // the effects on the operands and how the operation processes various failure modes.
  let summary = "Changes the callee of a call operation to the specified one";
  let description = [{
    For each `func.call` payload operation associated with the handle, changes its 
    callee to be the symbol whose name is provided as an attribute to this operation.

    Generates a silenceable failure if the operand is associated with payload operations 
    that are not `func.call`.
    Only reads the operand.
  }];

  // The arguments include the handle to the payload operations and the attribute that 
  // specifies the new callee. The handle must implement TransformHandleTypeInterface.   
  // We use a string attribute as the symbol may not exist in the transform IR so the 
  // verification may fail. 
  // 这里修改，限定ins是func.call
  let arguments = (ins 
    Transform_ConcreteOpType<"func.call">:$call,
    StrAttr:$new_target);

  // The results are empty as the transformation does not produce any new payload.
  let results = (outs);

  // Provide nice syntax.
  let assemblyFormat = "$call `,` $new_target attr-dict `:` type($call)";

  // 由于使用了eachonetrait，因此需要实现特定的方法
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::CallOp call,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}



#endif // MY_EXTENSION
