"""Small deterministic classifier for the challenge's simple PNG shapes."""

import base64
import binascii
from collections import deque
from io import BytesIO
from statistics import median
from typing import Literal

from PIL import Image, UnidentifiedImageError


Shape = Literal["rectangle", "triangle", "circle"]
MAX_IMAGE_BYTES = 5_000_000
MAX_PIXELS = 1_000_000
BACKGROUND_DISTANCE = 24


def _decode_png(image_base64: str) -> Image.Image:
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise ValueError("image_base64 must be a non-empty base64 PNG")

    encoded = image_base64.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or "base64" not in header.lower():
            raise ValueError("image data URI must contain base64 data")

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc

    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("PNG must be between 1 byte and 5 MB")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("decoded image must be a PNG")

    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("decoded image is not a readable PNG") from exc

    if image.width * image.height > MAX_PIXELS:
        raise ValueError("PNG must contain at most 1,000,000 pixels")

    # Transparency is composited onto white so transparent backgrounds behave
    # like the challenge's ordinary white image backgrounds.
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, "white")
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def _background_color(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    pixels = image.load()
    border = []

    for x in range(width):
        border.append(pixels[x, 0])
        if height > 1:
            border.append(pixels[x, height - 1])
    for y in range(1, height - 1):
        border.append(pixels[0, y])
        if width > 1:
            border.append(pixels[width - 1, y])

    return tuple(int(median(channel)) for channel in zip(*border))


def _largest_foreground_component(image: Image.Image) -> list[tuple[int, int]]:
    width, height = image.size
    pixels = image.load()
    background = _background_color(image)
    threshold_squared = BACKGROUND_DISTANCE**2
    foreground = bytearray(width * height)

    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            distance = (
                (red - background[0]) ** 2
                + (green - background[1]) ** 2
                + (blue - background[2]) ** 2
            )
            foreground[y * width + x] = distance >= threshold_squared

    visited = bytearray(width * height)
    largest: list[tuple[int, int]] = []
    for start, is_foreground in enumerate(foreground):
        if not is_foreground or visited[start]:
            continue

        visited[start] = 1
        queue = deque([start])
        component: list[tuple[int, int]] = []
        while queue:
            index = queue.popleft()
            x, y = index % width, index // width
            component.append((x, y))
            for nx, ny in (
                (x - 1, y - 1),
                (x, y - 1),
                (x + 1, y - 1),
                (x - 1, y),
                (x + 1, y),
                (x - 1, y + 1),
                (x, y + 1),
                (x + 1, y + 1),
            ):
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = ny * width + nx
                    if foreground[neighbor] and not visited[neighbor]:
                        visited[neighbor] = 1
                        queue.append(neighbor)

        if len(component) > len(largest):
            largest = component

    if len(largest) < 9:
        raise ValueError("PNG does not contain a recognizable shape")
    return largest


def identify_shape(image_base64: str) -> Shape:
    """Classify a filled or outlined axis-aligned rectangle, circle, or triangle."""

    component = _largest_foreground_component(_decode_png(image_base64))
    min_x = min(x for x, _ in component)
    max_x = max(x for x, _ in component)
    min_y = min(y for _, y in component)
    max_y = max(y for _, y in component)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # Fill the span between the leftmost and rightmost shape pixel on each
    # scanline. This measures the silhouette and works for filled and outlined
    # inputs: rectangles approach 1.0, circles pi/4, and triangles 0.5.
    row_bounds: dict[int, list[int]] = {}
    for x, y in component:
        bounds = row_bounds.setdefault(y, [x, x])
        bounds[0] = min(bounds[0], x)
        bounds[1] = max(bounds[1], x)

    silhouette_area = sum(right - left + 1 for left, right in row_bounds.values())
    fill_ratio = silhouette_area / (width * height)

    if fill_ratio >= 0.90:
        return "rectangle"
    if fill_ratio >= 0.64:
        return "circle"
    return "triangle"
