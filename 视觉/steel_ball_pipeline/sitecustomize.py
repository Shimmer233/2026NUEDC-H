"""Compatibility shims loaded by subprocesses without modifying the clean vendor checkout."""

from PIL import ImageFont


if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    def _getsize(self, text, *args, **kwargs):
        left, top, right, bottom = self.getbbox(text, *args, **kwargs)
        return right - left, bottom - top

    ImageFont.FreeTypeFont.getsize = _getsize
