add_mlir_library(DlyInferShape 
    ShapeInferencePass.cpp 
    TestDefUsePass.cpp
    DEPENDS 
    MLIRDlyIncGen 
    DlyShapeInferenceIncGen
)