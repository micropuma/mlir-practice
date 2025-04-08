// RUN: mlir-opt %s \
// RUN:   --pass-pipeline="builtin.module(transform-interpreter{ \
// RUN:        debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},\
// RUN:        canonicalize,cse,symbol-dce)" \
// RUN:   --split-input-file --verify-diagnostics

// ****************************** IMPORTANT NOTE ******************************
//
// If you are changing this file, you may also need to change
// mlir/docs/Tutorials/Transform accordingly.
//
// ****************************************************************************

// debug version
// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(
//       %arg0: !transform.any_op,
//       // expected-note @below {{handle to invalidated ops}}
//       %arg1: !transform.op<"linalg.matmul">,
//       %arg2: !transform.op<"linalg.elemwise_binary">) {
//     transform.debug.emit_remark_at %arg1, "matmul" : !transform.op<"linalg.matmul">
//     transform.debug.emit_remark_at %arg2, "elemwise_binary" : !transform.op<"linalg.elemwise_binary">
//     transform.yield
//   }
// }

// tiling versi
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      // expected-note @below {{handle to invalidated ops}}
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">) {

    // We can cast one type to another as long as operations are compatible
    // with both types. This creates "aliasing" handles.
    %casted = transform.cast %arg1 : !transform.op<"linalg.matmul">
        to !transform.any_op

    %loop, %tiled = transform.structured.tile_using_forall %arg1
                    tile_sizes [4, 32]
      : (!transform.op<"linalg.matmul">)
     -> (!transform.any_op, !transform.any_op)

    // Invalidate the original handle.
    // casted is an alias of %arg1, so it is also invalidated.
    // Consuming an operand invalidates the consumed handle and any other handle
    // that is associated with the same payload operations, or payload
    // operations nested in them.
    // transform.debug.emit_remark_at %casted, "remark"
    //   : !transform.any_op
    transform.yield
  }
}

// Original function to optimize.
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  // expected-note @below {{payload op}}
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>
  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}

