#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main(int argc, char **argv) {
    MLIRContext ctx;

    // 注册必要的dailect
    ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();
    // 读入必要的文件
    auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);
    // 输出dialect
    src->print(llvm::outs());
    src->dump();

    return 0;
}

