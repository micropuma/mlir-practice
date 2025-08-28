#!/bin/bash

../build/bin/dlyc ./codegen.toy --emit=mlir
../build/bin/dlyc ./canonicalize.toy --emit=mlir --opt
