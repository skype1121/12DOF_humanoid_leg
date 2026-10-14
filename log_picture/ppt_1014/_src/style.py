"""발표 자료 공통 스타일(색·글꼴)."""
from PIL import ImageFont

BG = (13, 16, 22)
CARD = (22, 27, 36)
CARD_LINE = (38, 45, 58)
TXT = (234, 238, 244)
SUB = (150, 160, 176)
DIM = (104, 114, 130)
GRID = (44, 52, 66)

GREEN = (86, 211, 140)
BLUE = (96, 165, 250)
VIOLET = (178, 148, 250)
AMBER = (245, 184, 82)
CORAL = (248, 122, 108)

_K = "/usr/share/fonts/opentype/noto/NotoSansCJK-%s.ttc"
_cache = {}


def font(size, weight="Regular"):
    key = (size, weight)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(_K % weight, size, index=1)   # index 1 = KR
    return _cache[key]


def mono(size, bold=False):
    key = (size, "mono", bold)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono%s.ttf" % ("-Bold" if bold else ""), size)
    return _cache[key]


def tw(draw, text, f):
    return draw.textlength(text, font=f)
