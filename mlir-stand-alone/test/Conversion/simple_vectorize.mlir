module {
  func.func @add(%0: i32, %1:i32) -> i32 {
    %result = arith.addi %0, %1 : i32
    return %result : i32
  }

  func.func @add_vector(%0: vector<4xi32>, %1: vector<4xi32>) -> vector<4xi32> {
    %result = arith.addi %0, %1 : vector<4xi32>
    return %result : vector<4xi32>
  }
  
  // 函数调用
  func.func @main() -> i32 {
    %a = arith.constant 10 : i32
    %b = arith.constant 20 : i32
    %sum = call @add(%a, %b) : (i32, i32) -> i32
    %c = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
    %d = arith.constant dense<[10, 20, 30, 40]> : vector<4xi32>
    %sum2 = call @add_vector(%c, %d) : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    return %sum : i32
  }
}