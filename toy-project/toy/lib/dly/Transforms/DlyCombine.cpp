#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "dly/Dialect.h"
using namespace mlir;
using namespace toy;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "dly/DlyCombine.h.inc"
} // namespace

// 转换matmul(matmul(A,B),C) -> matmul(A,matmul(B,C))
struct PermutateMatmul : public mlir::OpRewritePattern<MatmulOp> {
    PermutateMatmul(mlir::MLIRContext *context) : OpRewritePattern<MatmulOp>(context, /*benefit=*/1) {}

    mlir::LogicalResult matchAndRewrite(MatmulOp op, mlir::PatternRewriter &rewriter) const override {\
        // 获取matmul op的左右两个操作
        auto lhs = op.getOperand(0);
        auto rhs = op.getOperand(1);

        // 判断左边的op的是否是一个matmul的op
        auto matmul_lhs = llvm::dyn_cast<MatmulOp>(lhs.getDefiningOp());
        if (!matmul_lhs) {
            return failure();    // 提前终止，match and rewrite失败
        } else {
            // 生成 BxC
            MatmulOp BXC = rewriter.create<MatmulOp>(op.getLoc(), matmul_lhs.getOperand(1), rhs);
            // 生成 Ax(BxC)
            MatmulOp AXBC = rewriter.create<MatmulOp>(op.getLoc(), matmul_lhs.getOperand(0), BXC);
            rewriter.replaceOp(op, AXBC);
        }

        return success();
    }
}; 

/// write a single rewrite pattern to handle transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
    /// We register this pattern to match every toy.transpose in the IR.
    /// The "benefit" is used by the framework to order the patterns and process
    /// them in order of profitability.
    SimplifyRedundantTranspose(mlir::MLIRContext *context) : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

    /// This method is attempting to match a pattern and rewrite it. The rewriter
    /// argument is the orchestrator of the sequence of rewrites. It is expected
    /// to interact with it to perform any changes to the IR from here.
    mlir::LogicalResult matchAndRewrite(TransposeOp op, mlir::PatternRewriter &rewriter) const override {
        // Look through the input of the current transpose.
        mlir::Value transposeInput = op.getOperand();
        // Check if the input is also a transpose.
        // use getDefiningOp() to get the defining operation of the value.
        TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

        // Input defined by another transpose? If not, no match.
        if (!transposeInputOp)
            return failure();

        // Otherwise, we have a redundant transpose. Replace it by the input of
        // the current transpose operation.
        rewriter.replaceOp(op, {transposeInputOp.getOperand()});
        return success();
    }
};

/// Register out patterns for rewrite by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
    results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}

/// Register out patterns for rewrite by the Canonicalization framework.
void MatmulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
    results.add<PermutateMatmul>(context);
}