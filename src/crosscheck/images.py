"""Image checks without third-party packages."""

from __future__ import annotations

MAX_IMAGE = 20 * 1024 * 1024


class ImageError(ValueError):
    pass


def kind(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return None


def check_image(data: bytes) -> bytes:
    if not data:
        raise ImageError("no photo received")
    if len(data) > MAX_IMAGE:
        raise ImageError("the photo is too large (20 MB at most)")
    if kind(data) is None:
        raise ImageError("use a JPEG or PNG photo")
    return data
