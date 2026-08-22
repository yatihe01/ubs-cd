"""Pure implementations behind the Tool Box MCP tools."""

from typing import Literal

from challenges.tool_box.shape_recognition import identify_shape


def get_name() -> str:
    """Use when asked for your name or what you are called."""

    return "Nova"


def calculate(
    left: int,
    operator: Literal["+", "-", "*", "/"],
    right: int,
) -> int | float:
    """Calculate one arithmetic operation using integer operands from -100 to 100."""

    if not -100 <= left <= 100 or not -100 <= right <= 100:
        raise ValueError("operands must be integers between -100 and 100")

    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator != "/":
        raise ValueError("operator must be one of +, -, *, or /")
    if right == 0:
        raise ValueError("cannot divide by zero")

    quotient = left / right
    return int(quotient) if quotient.is_integer() else quotient


def recognize_shape(
    image_base64: str,
) -> Literal["rectangle", "triangle", "circle"]:
    """Identify a base64 PNG and return exactly rectangle, triangle, or circle."""

    return identify_shape(image_base64)
