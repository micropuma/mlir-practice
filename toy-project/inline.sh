#!/bin/bash

build/bin/dlyc -emit=mlir --opt test/toy/shape_inference.mlir
