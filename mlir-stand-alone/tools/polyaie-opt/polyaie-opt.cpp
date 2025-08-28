//===----------------------------------------------------------------------===//
//
// Copyright 2021-2021 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "polyaie/InitAllDialects.h"
#include "polyaie/InitAllPasses.h"

using namespace mlir;

int main(int argc, char **argv) {
  // used to register the pass.
  DialectRegistry registry;
  polyaie::registerAllDialects(registry);
  polyaie::registerAllPasses();

  return asMainReturnCode(MlirOptMain(argc, argv, "polyaie-opt", registry));
}
