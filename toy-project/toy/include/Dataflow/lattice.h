#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

struct MetadataLatticeValue {
  MetadataLatticeValue() = default;
  MetadataLatticeValue(DictionaryAttr attr) : metadata(attr.begin(), attr.end()) {}

  static MetadataLatticeValue getPessimisticValueState(MLIRContext *context) {
    return MetadataLatticeValue();
  }

  static MetadataLatticeValue getPessimisticValueState(Value value) {
    if (Operation *parentOp = value.getDefiningOp()) {
      if (auto metadata = parentOp->getAttrOfType<DictionaryAttr>("metadata"))
        return MetadataLatticeValue(metadata);
    }
    return MetadataLatticeValue();
  }

  static MetadataLatticeValue join(const MetadataLatticeValue &lhs, const MetadataLatticeValue &rhs) {
    MetadataLatticeValue result;
    for (const auto &lhsIt : lhs.metadata) {
      auto it = rhs.metadata.find(lhsIt.first);
      if (it != rhs.metadata.end() && it->second == lhsIt.second)
        result.metadata.insert(lhsIt);
    }
    return result;
  }

  bool operator==(const MetadataLatticeValue &rhs) const {
    if (metadata.size() != rhs.metadata.size())
      return false;
    for (const auto &it : metadata) {
      auto rhsIt = rhs.metadata.find(it.first);
      if (rhsIt == rhs.metadata.end() || it.second != rhsIt->second)
        return false;
    }
    return true;
  }

  DenseMap<StringAttr, Attribute> metadata;
};

