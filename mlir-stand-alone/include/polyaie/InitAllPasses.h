//===----------------------------------------------------------------------===//
//
// Copyright 2021-2021 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#ifndef POLYAIE_INITALLPASSES_H
#define POLYAIE_INITALLPASSES_H

#include "mlir/InitAllPasses.h"
#include "polyaie/Passes.h"

#include "aie/Conversion/Passes.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEVec/Analysis/Passes.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/InitialAllDialect.h"
#include "aie/Target/LLVMIR/Dialect/All.h"

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

namespace mlir {
namespace polyaie {

// Add all the related passes.
inline void registerAllPasses() {
  // mlir passes.
  mlir::registerAllPasses();

  // PolyAIE passes.
  polyaie::registerPolyAIEPasses();

  // AIE passes.
  xilinx::registerConversionPasses();
  xilinx::AIE::registerAIEPasses();
  xilinx::AIEX::registerAIEXPasses();
  xilinx::aievec::registerAIEVecAnalysisPasses();
  xilinx::aievec::registerAIEVecPasses();
  xilinx::aievec::registerAIEVecPipelines();
}

} // namespace polyaie
} // namespace mlir

#endif // POLYAIE_INITALLPASSES_H
