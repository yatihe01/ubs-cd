"""Pure implementations behind the Tool Box MCP tools."""

import ast
import math
from typing import Literal

from challenges.tool_box.shape_recognition import identify_shape


def get_name(question: str) -> str:
    """Return the exact final answer to the supplied name question."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    return "Nova"


def calculate(
    left: int,
    operator: Literal["+", "-", "*", "/"],
    right: int,
) -> int | float:
    """Calculate exactly one arithmetic operation using operands from -100 to 100."""

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


def evaluate_expression(expression: str) -> int | float:
    """Evaluate a complete arithmetic expression with standard precedence."""

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    if len(expression) > 200:
        raise ValueError("expression must contain at most 200 characters")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("expression is not valid arithmetic") from exc

    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError("expression is too complex")

    result = _evaluate_node(tree.body)
    if isinstance(result, float):
        if not math.isfinite(result):
            raise ValueError("expression result must be finite")
        return int(result) if result.is_integer() else result
    return result


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("operands must be integers")
        if not -100 <= value <= 100:
            raise ValueError("operands must be integers between -100 and 100")
        return value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if not isinstance(node, ast.BinOp):
        raise ValueError("expression may contain only numbers and +, -, *, /")

    left = _evaluate_node(node.left)
    right = _evaluate_node(node.right)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        if right == 0:
            raise ValueError("cannot divide by zero")
        return left / right
    raise ValueError("expression may contain only +, -, *, and /")


def recognize_shape(
    image_base64: str,
) -> Literal["rectangle", "triangle", "circle"]:
    """Identify a base64 PNG and return exactly rectangle, triangle, or circle."""

    return identify_shape(image_base64)
