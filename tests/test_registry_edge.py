"""Edge-case tests for ToolRegistry."""

import pytest
from laua.tools.registry import Tool, ToolRegistry


def _make_tool(name: str = "echo_tool"):
    async def handler(msg: str) -> str:
        return msg

    return Tool(
        name=name,
        description="Returns msg.",
        parameters_schema={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
        handler=handler,
    )


# ── dispatch: unknown tool ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_unknown_tool():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="Unknown tool"):
        await reg.dispatch("nonexistent_tool", {})


# ── dispatch: schema validation ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_missing_required_arg():
    reg = ToolRegistry()
    reg.register(_make_tool())
    with pytest.raises(ValueError, match="Invalid arguments"):
        await reg.dispatch("echo_tool", {})  # msg is required


@pytest.mark.asyncio
async def test_dispatch_wrong_type():
    reg = ToolRegistry()
    reg.register(_make_tool())
    with pytest.raises(ValueError, match="Invalid arguments"):
        await reg.dispatch("echo_tool", {"msg": 42})  # must be string


@pytest.mark.asyncio
async def test_dispatch_extra_field_causes_type_error():
    """
    Extra fields pass jsonschema (no additionalProperties: false) but then
    cause a TypeError in handler(**arguments). Ensures the caller is aware
    this can happen.
    """
    reg = ToolRegistry()
    reg.register(_make_tool())
    with pytest.raises(TypeError):
        await reg.dispatch("echo_tool", {"msg": "hi", "extra": "oops"})


@pytest.mark.asyncio
async def test_dispatch_valid_call():
    reg = ToolRegistry()
    reg.register(_make_tool())
    result = await reg.dispatch("echo_tool", {"msg": "hello"})
    assert result == "hello"


# ── register: overwrite ───────────────────────────────────────────────────────

def test_register_overwrite_silently_replaces():
    """Registering the same name twice replaces the first without error."""
    reg = ToolRegistry()

    async def v1() -> str:
        return "v1"

    async def v2() -> str:
        return "v2"

    reg.register(Tool("t", "v1", {"type": "object", "properties": {}}, v1))
    reg.register(Tool("t", "v2", {"type": "object", "properties": {}}, v2))
    assert reg.get("t").description == "v2"


# ── all_schemas ───────────────────────────────────────────────────────────────

def test_all_schemas_empty_registry():
    reg = ToolRegistry()
    assert reg.all_schemas() == []


def test_all_schemas_format():
    reg = ToolRegistry()
    reg.register(_make_tool("tool_a"))
    schemas = reg.all_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "tool_a"


# ── get ───────────────────────────────────────────────────────────────────────

def test_get_missing_returns_none():
    assert ToolRegistry().get("no_such_tool") is None


# ── handler raising ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handler_exception_propagates():
    async def boom(msg: str):
        raise RuntimeError("handler exploded")

    reg = ToolRegistry()
    reg.register(Tool(
        "boom_tool", "explodes",
        {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
        boom,
    ))
    with pytest.raises(RuntimeError, match="handler exploded"):
        await reg.dispatch("boom_tool", {"msg": "test"})
