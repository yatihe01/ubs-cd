import asyncio
import base64
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from challenges.tool_box.server import mcp
from challenges.tool_box.tools import (
    calculate,
    evaluate_expression,
    get_name,
    recognize_shape,
)


def make_shape(shape: str, *, outline: bool = False, data_uri: bool = False) -> str:
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    fill = None if outline else "black"
    width = 4 if outline else 1

    if shape == "rectangle":
        draw.rectangle((20, 25, 80, 75), fill=fill, outline="black", width=width)
    elif shape == "circle":
        draw.ellipse((20, 20, 80, 80), fill=fill, outline="black", width=width)
    elif shape == "triangle":
        draw.polygon(
            ((50, 15), (15, 85), (85, 85)),
            fill=fill,
            outline="black",
            width=width,
        )
    else:
        raise AssertionError(f"unsupported test shape: {shape}")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}" if data_uri else encoded


def test_name_is_valid():
    name = get_name("What is your name?")

    assert 3 <= len(name) <= 30
    assert all(character.isalnum() or character in " _-'" for character in name)


@pytest.mark.parametrize("question", ["", "   ", None])
def test_name_rejects_an_empty_question(question):
    with pytest.raises(ValueError, match="non-empty"):
        get_name(question)


def test_agent_sees_purpose_specific_tool_schemas():
    tools = asyncio.run(mcp.get_tools())

    assert list(tools) == [
        "answer_name_question",
        "evaluate_expression",
        "calculate",
        "recognize_shape",
        "retrieve_study_passages",
        "choose_next_node",
    ]
    assert "complete final answer" in tools["answer_name_question"].description
    assert tools["answer_name_question"].parameters["required"] == ["question"]
    assert tools["evaluate_expression"].parameters["required"] == ["expression"]
    assert "multiple operators" in tools["calculate"].description
    assert tools["retrieve_study_passages"].parameters["required"] == ["question"]
    assert tools["choose_next_node"].parameters["required"] == [
        "map_id",
        "current_node",
        "destination",
    ]


@pytest.mark.parametrize(
    ("left", "operator", "right", "expected"),
    [
        (2, "+", 2, 4),
        (2, "-", 5, -3),
        (-4, "*", 5, -20),
        (8, "/", 4, 2),
        (7, "/", 2, 3.5),
    ],
)
def test_calculate(left, operator, right, expected):
    assert calculate(left, operator, right) == expected


@pytest.mark.parametrize("operands", [(-101, "+", 0), (0, "+", 101)])
def test_calculate_rejects_out_of_range_operands(operands):
    with pytest.raises(ValueError, match="between -100 and 100"):
        calculate(*operands)


def test_calculate_rejects_division_by_zero():
    with pytest.raises(ValueError, match="divide by zero"):
        calculate(1, "/", 0)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 5", 17),
        ("-9 * 2 + 2", -16),
        ("(2 + 3) * 5", 25),
        ("7 / 2 + 1", 4.5),
        ("8 / (2 + 2)", 2),
    ],
)
def test_evaluate_expression_uses_standard_precedence(expression, expected):
    assert evaluate_expression(expression) == expected


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("", "non-empty"),
        ("101 + 1", "between -100 and 100"),
        ("2 ** 3", "may contain"),
        ("1 / 0", "divide by zero"),
        ("sum([1, 2])", "only numbers"),
    ],
)
def test_evaluate_expression_rejects_unsupported_input(expression, message):
    with pytest.raises(ValueError, match=message):
        evaluate_expression(expression)


@pytest.mark.parametrize("shape", ["rectangle", "circle", "triangle"])
@pytest.mark.parametrize("outline", [False, True])
def test_recognize_shape(shape, outline):
    assert recognize_shape(make_shape(shape, outline=outline)) == shape


def test_recognize_shape_accepts_data_uri():
    assert recognize_shape(make_shape("circle", data_uri=True)) == "circle"


@pytest.mark.parametrize("payload", ["", "not base64", base64.b64encode(b"nope").decode()])
def test_recognize_shape_rejects_invalid_input(payload):
    with pytest.raises(ValueError):
        recognize_shape(payload)
