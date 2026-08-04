"""Parses .tflite flatbuffer files into mcufit's domain model."""

from __future__ import annotations

from pathlib import Path

import tflite

from ..domain.model import LayerInfo, ModelInfo, Quantization, TensorInfo
from .base import ModelParseError

_DTYPE_SIZES: dict[int, tuple[str, int]] = {
    tflite.TensorType.FLOAT32: ("float32", 4),
    tflite.TensorType.FLOAT16: ("float16", 2),
    tflite.TensorType.INT32: ("int32", 4),
    tflite.TensorType.UINT8: ("uint8", 1),
    tflite.TensorType.INT64: ("int64", 8),
    tflite.TensorType.INT16: ("int16", 2),
    tflite.TensorType.INT8: ("int8", 1),
    tflite.TensorType.BOOL: ("bool", 1),
    tflite.TensorType.FLOAT64: ("float64", 8),
    tflite.TensorType.UINT16: ("uint16", 2),
    tflite.TensorType.UINT32: ("uint32", 4),
    tflite.TensorType.UINT64: ("uint64", 8),
}

_OP_NAMES: dict[int, str] = {
    value: name
    for name, value in vars(tflite.BuiltinOperator).items()
    if not name.startswith("_") and isinstance(value, int)
}


class TFLiteModelParser:
    """ModelParser implementation for TensorFlow Lite flatbuffers."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in {".tflite", ".lite"}

    def parse(self, path: Path) -> ModelInfo:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ModelParseError(f"cannot read {path}: {exc}") from exc

        # Flatbuffers reads lazily, so corruption surfaces on any access,
        # not just on the initial wrap - hence the wide try.
        try:
            model = tflite.Model.GetRootAsModel(data, 0)
            if model.SubgraphsLength() == 0:
                raise ModelParseError(f"{path} contains no subgraphs")
            subgraph = model.Subgraphs(0)
            tensors = tuple(
                self._parse_tensor(model, subgraph, i)
                for i in range(subgraph.TensorsLength())
            )
            layers = tuple(
                self._parse_layer(model, subgraph, i)
                for i in range(subgraph.OperatorsLength())
            )
            graph_inputs = tuple(int(x) for x in subgraph.InputsAsNumpy())
            graph_outputs = tuple(int(x) for x in subgraph.OutputsAsNumpy())
        except ModelParseError:
            raise
        except Exception as exc:
            raise ModelParseError(f"{path} is not a valid .tflite file") from exc

        return ModelInfo(
            path=path,
            file_size_bytes=len(data),
            quantization=self._infer_quantization(tensors),
            tensors=tensors,
            layers=layers,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
        )

    def _parse_tensor(self, model, subgraph, index: int) -> TensorInfo:
        tensor = subgraph.Tensors(index)
        dtype_name, dtype_size = _DTYPE_SIZES.get(tensor.Type(), ("unknown", 4))

        shape: tuple[int, ...] = ()
        if tensor.ShapeLength() > 0:
            # A dynamic batch dimension (-1) is treated as 1: on a
            # microcontroller you run one inference at a time.
            shape = tuple(max(1, int(d)) for d in tensor.ShapeAsNumpy())

        elements = 1
        for dim in shape:
            elements *= dim

        buffer = model.Buffers(tensor.Buffer())
        has_data = buffer is not None and buffer.DataLength() > 0

        name = tensor.Name().decode("utf-8", "replace") if tensor.Name() else f"tensor_{index}"
        return TensorInfo(
            index=index,
            name=name,
            shape=shape,
            dtype=dtype_name,
            size_bytes=elements * dtype_size,
            is_weight=has_data,
        )

    def _parse_layer(self, model, subgraph, index: int) -> LayerInfo:
        op = subgraph.Operators(index)
        opcode = model.OperatorCodes(op.OpcodeIndex())
        # Newer schemas moved builtin codes past 127; older files use the
        # deprecated field. Taking the max yields the right one in both cases.
        code = max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())
        inputs = tuple(int(x) for x in op.InputsAsNumpy() if int(x) >= 0)
        outputs = tuple(int(x) for x in op.OutputsAsNumpy() if int(x) >= 0)
        return LayerInfo(
            index=index,
            op_name=_OP_NAMES.get(code, f"OP_{code}"),
            input_tensors=inputs,
            output_tensors=outputs,
        )

    @staticmethod
    def _infer_quantization(tensors: tuple[TensorInfo, ...]) -> Quantization:
        weight_dtypes = {t.dtype for t in tensors if t.is_weight}
        # Bias tensors are int32 in int8-quantized models; ignore them.
        weight_dtypes.discard("int32")
        if weight_dtypes == {"int8"} or weight_dtypes == {"int8", "uint8"}:
            return Quantization.INT8
        if weight_dtypes == {"float32"}:
            return Quantization.FLOAT32
        if weight_dtypes == {"float16"}:
            return Quantization.FLOAT16
        if len(weight_dtypes) > 1:
            return Quantization.MIXED
        return Quantization.UNKNOWN
