# Shared pytest fixtures and helpers.

from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from its.llm import ChatModel


def as_chat_model(double: object) -> "ChatModel":
    return cast("ChatModel", double)


MATH_KB: dict[str, str] = {
    "the quadratic formula":
    "For ax^2 + bx + c = 0 with a != 0, the solutions are x = (-b +/- sqrt(b^2 - 4ac)) / (2a). "
    "The quantity b^2 - 4ac is the discriminant: positive gives two real roots, zero gives "
    "one repeated root, negative gives two complex roots.",
    "the pythagorean theorem":
    "In a right triangle with legs a and b and hypotenuse c (the side opposite the right "
    "angle), a^2 + b^2 = c^2. It holds only for right triangles.",
    "completing the square":
    "To complete the square for x^2 + bx, add and subtract (b/2)^2, giving (x + b/2)^2 - (b/2)^2. "
    "This rewrites a quadratic in vertex form so its minimum or maximum can be read off directly.",
    "the difference of squares":
    "A difference of two squares factors as a^2 - b^2 = (a - b)(a + b). Note a^2 + b^2 does NOT "
    "factor over the real numbers.",
    "the slope of a line":
    "The slope between two points (x1, y1) and (x2, y2) is m = (y2 - y1) / (x2 - x1), the rise "
    "over the run. Horizontal lines have slope 0 and vertical lines have undefined slope.",
    "the laws of exponents":
    "For the same base: a^m * a^n = a^(m+n), a^m / a^n = a^(m-n), and (a^m)^n = a^(m*n). "
    "Also a^0 = 1 for a != 0, and a^(-n) = 1 / a^n.",
    "the area of a circle":
    "A circle of radius r has area A = pi * r^2 and circumference C = 2 * pi * r. The diameter "
    "is twice the radius, d = 2r.",
}


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    from its.config import SIMULATOR_MODEL_ID
    return AutoTokenizer.from_pretrained(SIMULATOR_MODEL_ID)
