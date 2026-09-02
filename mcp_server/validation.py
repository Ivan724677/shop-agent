"""Small deterministic JSON Schema subset used at the tool trust boundary."""

from __future__ import annotations

import re
from typing import Any


class SchemaValidationError(ValueError):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def validate_json(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    if not schema:
        return
    expected = schema.get("type")
    if expected and not _matches_type(value, expected):
        raise SchemaValidationError(path, f"期望 {expected}，实际为 {type(value).__name__}")
    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(path, f"必须等于 {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(path, f"必须是 {schema['enum']} 之一")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise SchemaValidationError(path, "缺少必填字段：" + ", ".join(missing))
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                raise SchemaValidationError(path, "包含未声明字段：" + ", ".join(unexpected))
        for key, item in value.items():
            if key in properties:
                validate_json(item, properties[key], f"{path}.{key}")

    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise SchemaValidationError(path, f"至少需要 {schema['minItems']} 项")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise SchemaValidationError(path, f"最多允许 {schema['maxItems']} 项")
        if schema.get("uniqueItems") and len({_stable(item) for item in value}) != len(value):
            raise SchemaValidationError(path, "数组项必须唯一")
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            validate_json(item, item_schema, f"{path}[{index}]")

    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise SchemaValidationError(path, f"长度不能小于 {schema['minLength']}")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise SchemaValidationError(path, f"长度不能超过 {schema['maxLength']}")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise SchemaValidationError(path, "格式不符合约束")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(path, f"不能小于 {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise SchemaValidationError(path, f"必须大于 {schema['exclusiveMinimum']}")


def _matches_type(value: Any, expected: str) -> bool:
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "boolean": lambda: isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "null": lambda: value is None,
    }
    return checks.get(expected, lambda: True)()


def _stable(value: Any) -> str:
    if isinstance(value, dict):
        return repr(sorted((key, _stable(item)) for key, item in value.items()))
    if isinstance(value, list):
        return repr([_stable(item) for item in value])
    return repr(value)
