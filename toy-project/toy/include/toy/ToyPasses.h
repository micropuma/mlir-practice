#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "toy/ToyOps.h"
#include <memory>
#include <mutex>

namespace toy {
// 生成定义
#define GEN_PASS_DECL
#include "toy/ToyPasses.h.inc"

// 写create函数
std::unique_ptr<mlir::Pass> createConvertToyToArithPass(ConvertToyToArithOptions options={});
std::unique_ptr<mlir::Pass> createDCEPass();

// 生成注册函数
#define GEN_PASS_REGISTRATION
#include "toy/ToyPasses.h.inc"

}