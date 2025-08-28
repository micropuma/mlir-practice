//===----------------------------------------------------------------------===//
//
// Copyright 2021-2021 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "polyaie/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/LogicalResult.h"

using namespace llvm;
using namespace mlir;
using namespace polyaie;

namespace {
struct AffinePreprocess : public AffinePreprocessBase<AffinePreprocess> {
  void runOnOperation() override;
};
} // namespace

// ============= some preprocessing methods target at AIE architecture ====================
/// return the funOp with certain name: topFunName
static auto getTopFunc(Operation* op, StringRef topFuncName) -> func::FuncOp {
  // iterate over all the ops in module.
  for (auto &region: op->getRegions()) {
    for (auto &block : region.getBlocks()) {
      for (auto func: block.getOps<func::FuncOp>()) {
        // only process FuncOp
        llvm::errs() << "Found function: " << func.getName() << "\n";

        if (func.getName().str() == topFuncName) {
          return func;
        }
      }
    }
  }

  emitError(op->getLoc(), "doesn't contain function " + topFuncName);
  return nullptr;
} 

/// erase unused arguments in function
static auto eraseConstantArguments(func::FuncOp topFunc) -> void {
  // initialize bitvector to all false.
  unsigned sz = topFunc.getNumArguments();
  BitVector argsToErase(sz, false);

  // iterate over arguments of funcOp
  for (unsigned int i = 0; i < sz; i++) {
    // judge that arg if it is a constant one.
    if (topFunc.getArgAttr(i, "scop.constant_value")) {
      argsToErase.set(i);
    }
  }

  // clear all unused args
  topFunc.eraseArguments(argsToErase);
}

/// Unroll all the loop forms
static auto unrollAllLoops(func::FuncOp topFunc) -> LogicalResult {
  SmallVector<affine::AffineForOp, 16> workList;

  // Recursively traverse all the regions and blocks nested inside the function
  // and apply the callback on every single operation in post-order.
  topFunc.walk([&](affine::AffineForOp op){
    // process AffineForOp
    workList.push_back(op);
  }); 

  // iterate overall the affine op in post-order to unroll it.
  for (auto loop : workList) {
    if (llvm::failed(loopUnrollFull(loop))) {
      return failure();
    }
  }

  return success();
}

/// simplify all the funOp, support canonicalization and affineStructuresPass
static auto simplifyFunc(func::FuncOp topFunc) -> LogicalResult {
  PassManager pm(topFunc->getContext(), "func.func");
  pm.addPass(createCanonicalizerPass());
  pm.addPass(affine::createSimplifyAffineStructuresPass());

  return pm.run(topFunc);
}

/// duplicate all the subfunctions in topFunction
static auto duplicateSubFuncs(func::FuncOp func) {
  auto builder = OpBuilder(func);

  // iterate and collect all the CallOp in topFunc.
  SmallVector<func::CallOp, 16> callList;
  func.walk([&](func::CallOp call) { callList.push_back(call); });

  unsigned coreIdx = 0;
  for (auto call : callList) {
    // search certain calleeOp function, in module nest funcOp.
    auto calleeOp = SymbolTable::lookupSymbolIn(
        func->getParentOfType<ModuleOp>(), call.getCallee());

    if (!calleeOp) {
      func->emitError() << "Callee not found: " << call.getCallee();
      return; 
    }

    auto callee = dyn_cast<func::FuncOp>(calleeOp);

    // Create a new callee function for each call operation.
    builder.setInsertionPointAfter(callee);
    auto newCallee = callee.clone();
    builder.insert(newCallee);

    // Set up a new name.
    auto newName = call.getCallee().str() + "_AIE" + std::to_string(coreIdx);
    newCallee.setName(newName);
    // update the callee attr, to point to the new callee
    call->setAttr("callee", SymbolRefAttr::get(builder.getContext(), newName));

    // Localize the constant into the new callee.
    auto argsToErase = llvm::BitVector();
    auto operandsToErase = llvm::BitVector();
    unsigned int sz = call->getNumOperands();
    for (unsigned i = 0; i < sz; ++i) {
      // check if the operand is a constantop
      if (auto param = call.getOperand(i).getDefiningOp<arith::ConstantOp>()) {
        // copy the declaration point of this operand into the funciton
        // and remove the function argument
        builder.setInsertionPointToStart(&newCallee.front());
        auto newParam = param.clone();
        builder.insert(newParam);

        newCallee.getArgument(i).replaceAllUsesWith(newParam);

        argsToErase.push_back(true);
        operandsToErase.push_back(true);
        continue;
      }
      argsToErase.push_back(false);
      operandsToErase.push_back(false);
    }

    // erase the useless operands and args
    newCallee.eraseArguments(argsToErase);
    call->eraseOperands(operandsToErase);

    // Move to the next AIE call.
    ++coreIdx;
  }
}

// static mlir::Value materializeLoad(OpBuilder &b, mlir::Type type, ValueRange inputs,
//                              Location loc) {
//   assert(inputs.size() == 1);
//   auto inputType = inputs[0].getType().dyn_cast<MemRefType>();
//   assert(inputType && inputType.getElementType() == type);
//   return b.create<affine::AffineLoadOp>(loc, inputs[0], ValueRange({}));
// }

// static mlir::Value materializeAllocAndStore(OpBuilder &b, MemRefType type,
//                                       ValueRange inputs, Location loc) {
//   assert(inputs.size() == 1);
//   assert(type.getElementType() == inputs[0].getType());
//   auto memref = b.create<memref::AllocOp>(loc, type);
//   b.create<affine::AffineStoreOp>(loc, inputs[0], memref, ValueRange({}));
//   return memref;
// }

// class ScalarBufferizeTypeConverter : public TypeConverter {
// public:
//   ScalarBufferizeTypeConverter() {
//     // Convert all scalar to memref.
//     addConversion([](mlir::Type type) -> mlir::Type {
//       if (isa<MemRefType>(type))
//         return MemRefType::get({}, type);
//       return type;
//     });
//     // Load the original scalar from memref.
//     addArgumentMaterialization(materializeLoad);
//     addSourceMaterialization(materializeLoad);
//     addTargetMaterialization(materializeAllocAndStore);
//   }
// };

// /// make all scalars bufferized. cause there is no registers or sort of that in AIE.
// static auto bufferizeAllScalars(Operation* op) -> LogicalResult {
// ScalarBufferizeTypeConverter typeConverter;
//   RewritePatternSet patterns(op->getContext());
//   ConversionTarget target(*op->getContext());

//   populateFuncOpTypeConversionPattern(patterns, typeConverter);
//   target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
//     return typeConverter.isSignatureLegal(op.getFunctionType()) &&
//            typeConverter.isLegal(&op.getBody());
//   });
//   populateCallOpTypeConversionPattern(patterns, typeConverter);
//   target.addDynamicallyLegalOp<func::CallOp>(
//       [&](func::CallOp op) { return typeConverter.isLegal(op); });
//   populateReturnOpTypeConversionPattern(patterns, typeConverter);

//   target.markUnknownOpDynamicallyLegal([&](Operation *op) {
//     return std::optional<bool>(isNotBranchOpInterfaceOrReturnLikeOp(op) || isLegalForReturnOpTypeConversionPattern(op, typeConverter));
//   });

//   return applyFullConversion(op, target, std::move(patterns));
// }

/// overide runOnOperation()
auto AffinePreprocess::runOnOperation() -> void {
  Operation* op = getOperation();

  // get the certain function determined by pass driver
  auto topFunc = getTopFunc(op, topFuncName);
  if (!topFunc) {
    // Signal that some invariant was broken when running. The IR is allowed to
    // be in an invalid state.
    signalPassFailure();
    return;
  }

  // Erase constant arguments of the top function. These constant are unused
  // and dangling there after the compilation.
  eraseConstantArguments(topFunc);

  // Unroll all loops in the top function.
  if (failed(unrollAllLoops(topFunc))) {
    signalPassFailure();
    return;
  }

  // Simplify the top function.
  if (failed(simplifyFunc(topFunc))) {
    signalPassFailure();
    return;
  }

  // Create a seperate function for each call in the top function.
  duplicateSubFuncs(topFunc);

  // // Bufferize all scalar arguments, which is quite common in accelerator like AIE.
  // if (failed(bufferizeAllScalars(op))) {
  //   signalPassFailure();
  // }
}

/*
lamda version is also allowed:
void registerMyPass() {
  PassRegistration<MyParametricPass>(
    []() -> std::unique_ptr<Pass> {
      std::unique_ptr<Pass> p = std::make_unique<MyParametricPass>();
      ;
      return p;
    });
}
*/
std::unique_ptr<Pass> polyaie::createAffinePreprocessPass() {
  return std::make_unique<AffinePreprocess>();
}
