module {
  func.func private @orig()
  func.func private @updated()
  func.func @test1() {
    call @updated() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.call">
      transform.my.change_call_target %0, "updated" : !transform.op<"func.call">
      transform.yield 
    }
  }
}

// -----
module {
  func.func private @orig()
  func.func @test2() {
    "my.mm4"() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.my.call_op_interface
      %1 = transform.my.call_to_op %0 : (!transform.my.call_op_interface) -> !transform.any_op
      transform.yield 
    }
  }
}

