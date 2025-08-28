#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/Function.h"

using namespace mlir;

class MetadataDataflowAnalysis : public PassWrapper<MetadataDataflowAnalysis, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    // 执行分析逻辑
    FuncOp func = getOperation();
    for (auto &block : func) {
      for (auto &op : block) {
        // 在这里进行对每个操作的分析
        MetadataLatticeValue latticeValue = MetadataLatticeValue::getPessimisticValueState(op.getResult(0));
        // 处理分析结果
      }
    }
  }
};

