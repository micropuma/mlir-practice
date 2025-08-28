module {
  func.func @add(%0: i32, %1:i32) -> i32 {
    %result = arith.addi %0, %1 : i32
    return %result : i32
  }
  
  // 函数调用
  func.func @main() -> i32 {
    %a = arith.constant 10 : i32
    %b = arith.constant 20 : i32
    %sum = call @add(%a, %b) : (i32, i32) -> i32
    return %sum : i32
  }
}

