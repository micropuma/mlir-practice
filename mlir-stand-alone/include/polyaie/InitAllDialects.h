//===----------------------------------------------------------------------===//
//
// Copyright 2021-2021 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#ifndef POLYAIE_INITALLDIALECTS_H
#define POLYAIE_INITALLDIALECTS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Targets/AIERT.h"

namespace mlir {
namespace polyaie {

// Add all the related dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    mlir::memref::MemRefDialect,
    mlir::affine::AffineDialect,
    mlir::func::FuncDialect,
    mlir::arith::ArithDialect,
    mlir::vector::VectorDialect,
    mlir::scf::SCFDialect,
    mlir::LLVM::LLVMDialect,
    xilinx::AIE::AIEDialect
  >();
  // clang-format on
}

} // namespace polyaie
} // namespace mlir

#endif // POLYAIE_INITALLDIALECTS_H
