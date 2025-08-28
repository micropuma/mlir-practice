// RUN: dlyc %s -emit=mlir -opt 2>&1 | FileCheck-10 %s

// Check the result of inlining+shape inference on an input module.

dly.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = dly.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = dly.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = dly.mul %0, %1 : tensor<*xf64>
  dly.return %2 : tensor<*xf64>
}
dly.func @main() {
  %0 = dly.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = dly.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = dly.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
  %3 = dly.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
  %4 = dly.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = dly.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  dly.print %5 : tensor<*xf64>
  dly.return
}

// CHECK-NOT: dly.func @multiply_transpose
// CHECK-NOT: tensor<*xf64>

// CHECK-LABEL: dly.func @main()
// CHECK:         [[VAL_0:%.*]] = dly.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
// CHECK:         [[VAL_1:%.*]] = dly.transpose([[VAL_0]] : tensor<2x3xf64>) to tensor<3x2xf64>
// CHECK:         [[VAL_2:%.*]] = dly.mul [[VAL_1]], [[VAL_1]] : tensor<3x2xf64>
// CHECK:         dly.print [[VAL_2]] : tensor<3x2xf64>
// CHECK:         dly.return
