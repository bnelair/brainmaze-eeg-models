"""Generate tests/data/tiny_affine.onnx, a 2-input/2-output toy model for the runtime tests.

    y = 2 * a + 1            a: float32 [batch, 3]
    s = sum(b, axis=-1)      b: float32 [batch, 4]  ->  s: [batch]

Needs the `onnx` package (dev only, not a dependency): ``pip install onnx``, then run this
script from the repository root. The generated file is committed (tests are never shipped).
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

a = helper.make_tensor_value_info("a", TensorProto.FLOAT, ["batch", 3])
b = helper.make_tensor_value_info("b", TensorProto.FLOAT, ["batch", 4])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])
s = helper.make_tensor_value_info("s", TensorProto.FLOAT, ["batch"])
two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two")
one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one")
axes = numpy_helper.from_array(np.array([-1], dtype=np.int64), "axes")
nodes = [
    helper.make_node("Mul", ["a", "two"], ["a2"]),
    helper.make_node("Add", ["a2", "one"], ["y"]),
    helper.make_node("ReduceSum", ["b", "axes"], ["s"], keepdims=0),
]
graph = helper.make_graph(nodes, "tiny_affine", [a, b], [y, s], [two, one, axes])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="brainmaze-tests")
model.ir_version = 8  # loadable by onnxruntime >= 1.19
onnx.checker.check_model(model)
onnx.save(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_affine.onnx"))
