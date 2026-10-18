"""JSON Schema validation and default value application for tester configs."""

from typing import Any


class SchemaValidationError(ValueError):
    """Raised when a configuration dictionary fails JSON Schema validation."""
    pass


def apply_schema_defaults(data: dict[str, Any] | None, schema: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with default values filled in from the JSON Schema."""
    if not isinstance(schema, dict):
        return data if isinstance(data, dict) else {}

    result = dict(data) if isinstance(data, dict) else {}
    properties = schema.get("properties", {})

    if isinstance(properties, dict):
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                if prop_name not in result:
                    if "default" in prop_schema:
                        result[prop_name] = prop_schema["default"]
                elif isinstance(result[prop_name], dict) and prop_schema.get("type") == "object":
                    result[prop_name] = apply_schema_defaults(result[prop_name], prop_schema)

    return result


def validate_schema(data: Any, schema: dict[str, Any], path: str = "config") -> None:
    """Validate data against a JSON Schema subset.
    
    Supported constraints:
    - type: "object", "array", "string", "number", "integer", "boolean", "null"
    - required: list of required property names
    - properties: child property schemas
    - enum: list of valid values
    - minimum, maximum: numeric boundaries
    - minLength, maxLength: string length boundaries
    - items: array item schema
    """
    if not isinstance(schema, dict):
        return

    # Check oneOf / anyOf composite schemas
    subschemas = schema.get("oneOf") or schema.get("anyOf")
    if isinstance(subschemas, list) and subschemas:
        last_specific_error: SchemaValidationError | None = None
        for sub in subschemas:
            if not isinstance(sub, dict):
                continue
            try:
                validate_schema(data, sub, path=path)
                return
            except SchemaValidationError as err:
                sub_type = sub.get("type")
                if (
                    (sub_type == "string" and isinstance(data, str))
                    or (sub_type == "array" and isinstance(data, list))
                    or (sub_type == "object" and isinstance(data, dict))
                ):
                    last_specific_error = err
        if last_specific_error is not None:
            raise last_specific_error
        allowed_types = [s.get("type") for s in subschemas if isinstance(s, dict) and s.get("type")]
        raise SchemaValidationError(
            f"Field '{path}' expected type {' or '.join(repr(t) for t in allowed_types)}, got '{type(data).__name__}'"
        )

    # Check type
    expected_type = schema.get("type")
    if expected_type:
        type_checks = {
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "null": lambda v: v is None,
        }
        checker = type_checks.get(expected_type)
        if checker and not checker(data):
            raise SchemaValidationError(
                f"Field '{path}' expected type '{expected_type}', got '{type(data).__name__}'"
            )

    # Check enum
    if "enum" in schema and isinstance(schema["enum"], list):
        if data not in schema["enum"]:
            raise SchemaValidationError(
                f"Field '{path}' value {data!r} is not one of allowed enum values: {schema['enum']}"
            )

    # Check number constraints
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            raise SchemaValidationError(
                f"Field '{path}' value {data} is less than minimum {schema['minimum']}"
            )
        if "maximum" in schema and data > schema["maximum"]:
            raise SchemaValidationError(
                f"Field '{path}' value {data} is greater than maximum {schema['maximum']}"
            )

    # Check string constraints
    if isinstance(data, str):
        if "minLength" in schema and len(data) < schema["minLength"]:
            raise SchemaValidationError(
                f"Field '{path}' length {len(data)} is shorter than minLength {schema['minLength']}"
            )
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            raise SchemaValidationError(
                f"Field '{path}' length {len(data)} is longer than maxLength {schema['maxLength']}"
            )

    # Check object constraints
    if isinstance(data, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for req_key in required:
                if req_key not in data:
                    raise SchemaValidationError(f"Field '{path}' is missing required property '{req_key}'")

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, val in data.items():
                if key in properties and isinstance(properties[key], dict):
                    validate_schema(val, properties[key], path=f"{path}.{key}")

    # Check array constraints
    if isinstance(data, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for idx, item in enumerate(data):
                validate_schema(item, items_schema, path=f"{path}[{idx}]")
