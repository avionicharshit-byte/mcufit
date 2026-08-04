"""Parses ONNX models. Requires the optional dependency: pip install mcufit[onnx]

Arena semantics differ between runtimes (onnxruntime vs TFLM), so ONNX
verdicts are estimates of activation memory pressure, not TFLM-exact
figures; exact mode stays .tflite-only.
"""

from __future__ import annotations

from pathlib import Path

from ..domain.model import LayerInfo, ModelInfo, Quantization, TensorInfo
from .base import ModelParseError

_DTYPE_SIZES = {
    1: ("float32", 4), 2: ("uint8", 1), 3: ("int8", 1), 4: ("uint16", 2),
    5: ("int16", 2), 6: ("int32", 4), 7: ("int64", 8), 9: ("bool", 1),
    10: ("float16", 2), 11: ("float64", 8), 12: ("uint32", 4), 13: ("uint64", 8),
}


class OnnxModelParser:
    """ModelParser implementation for ONNX graphs."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".onnx"

    def parse(self, path: Path) -> ModelInfo:
        try:
            import onnx
            from onnx import shape_inference
        except ImportError as exc:
            raise ModelParseError(
                "ONNX support needs the onnx package: pip install 'mcufit[onnx]'"
            ) from exc

        try:
            model = onnx.load(str(path))
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            raise ModelParseError(f"{path} is not a valid ONNX model: {exc}") from exc

        graph = model.graph
        initializer_names = {init.name for init in graph.initializer}

        tensors: list[TensorInfo] = []
        index_of: dict[str, int] = {}

        def register(name: str, shape, dtype_code: int, is_weight: bool) -> int:
            if name in index_of:
                return index_of[name]
            dtype, dtype_size = _DTYPE_SIZES.get(dtype_code, ("unknown", 4))
            elements = 1
            for dim in shape:
                elements *= max(1, dim)
            index = len(tensors)
            tensors.append(TensorInfo(
                index=index, name=name, shape=tuple(max(1, d) for d in shape),
                dtype=dtype, size_bytes=elements * dtype_size, is_weight=is_weight,
            ))
            index_of[name] = index
            return index

        for init in graph.initializer:
            register(init.name, tuple(init.dims), init.data_type, is_weight=True)

        for value in list(graph.input) + list(graph.value_info) + list(graph.output):
            if value.name in initializer_names:
                continue
            tensor_type = value.type.tensor_type
            shape = tuple(
                d.dim_value if d.HasField("dim_value") else 1
                for d in tensor_type.shape.dim
            )
            register(value.name, shape, tensor_type.elem_type, is_weight=False)

        layers: list[LayerInfo] = []
        for i, node in enumerate(graph.node):
            inputs = tuple(
                index_of[name] if name in index_of else register(name, (), 1, False)
                for name in node.input if name
            )
            outputs = tuple(
                index_of[name] if name in index_of else register(name, (), 1, False)
                for name in node.output if name
            )
            layers.append(LayerInfo(index=i, op_name=node.op_type.upper(), input_tensors=inputs, output_tensors=outputs))

        graph_inputs = tuple(
            index_of[v.name] for v in graph.input if v.name in index_of and v.name not in initializer_names
        )
        graph_outputs = tuple(index_of[v.name] for v in graph.output if v.name in index_of)

        all_tensors = tuple(tensors)
        return ModelInfo(
            path=path,
            file_size_bytes=path.stat().st_size,
            quantization=self._infer_quantization(all_tensors),
            tensors=all_tensors,
            layers=tuple(layers),
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
        )

    @staticmethod
    def _infer_quantization(tensors: tuple[TensorInfo, ...]) -> Quantization:
        weight_dtypes = {t.dtype for t in tensors if t.is_weight}
        weight_dtypes.discard("int32")
        weight_dtypes.discard("int64")
        if weight_dtypes <= {"int8", "uint8"} and weight_dtypes:
            return Quantization.INT8
        if weight_dtypes == {"float32"}:
            return Quantization.FLOAT32
        if weight_dtypes == {"float16"}:
            return Quantization.FLOAT16
        if len(weight_dtypes) > 1:
            return Quantization.MIXED
        return Quantization.UNKNOWN
