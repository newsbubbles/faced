"""Headless renderer: face params -> a standalone SVG string.

Mirrors the geometry in web/face.js so a face can be rendered with no browser
(useful for reports, RunPod runs, and environments where screenshots are blocked).
Feed it either raw face params or emotion meters (via FaceMapper).
"""
from __future__ import annotations

from .faceparams import FaceMapper

EYE = {"L": (150, 182), "R": (270, 182), "rx": 34, "ry": 24}
BROW = {"baseY": 140, "L": (186, 116), "R": (234, 304)}   # (inner, outer)
MOUTH = {"cx": 210, "cy": 302, "xL": 164, "xR": 256}
SKIN, SKIN2 = "#e8b48c", "#d79f77"


def _g(p, k):
    return float(p.get(k, 0.0))


def _brow(side, p):
    base = BROW["baseY"]
    inner, outer = BROW[side]
    s = 1 if side == "L" else -1
    innerY = base - _g(p, "brow_inner") * 16 + _g(p, "brow_furrow") * 11 + s * _g(p, "asymmetry") * 7
    outerY = base - _g(p, "brow_outer") * 15
    innerX = inner + s * _g(p, "brow_furrow") * 7
    midX = (innerX + outer) / 2
    midY = min(innerY, outerY) - 7 - _g(p, "brow_outer") * 3
    return f"M {outer} {outerY:.1f} Q {midX:.1f} {midY:.1f} {innerX:.1f} {innerY:.1f}"


def _lids(side, p):
    ex, ey = EYE[side]
    rx, ry = EYE["rx"], EYE["ry"]
    top, bot, x0, x1 = ey - ry, ey + ry, ex - rx - 2, ex + rx + 2
    openU = max(0.02, min(1.5, 1 + _g(p, "lid_upper")))
    cover = max(0.0, 1 - openU)
    lidY = top + cover * 2 * ry
    up = f"M {x0} {top-34} L {x1} {top-34} L {x1} {lidY:.1f} Q {ex} {lidY+9:.1f} {x0} {lidY:.1f} Z"
    lo = max(0.0, _g(p, "lid_lower_tense")) * 0.55
    loY = bot - lo * 2 * ry
    low = f"M {x0} {bot+34} L {x1} {bot+34} L {x1} {loY:.1f} Q {ex} {loY-7:.1f} {x0} {loY:.1f} Z"
    return up, low


def _mouth(p):
    cx, cy, xL, xR = MOUTH["cx"], MOUTH["cy"], MOUTH["xL"], MOUTH["xR"]
    curve = _g(p, "mouth_curve")
    op = max(0.0, _g(p, "mouth_open"))
    cornerY = cy - curve * 10
    topMidY = cy - curve * 17 - op * 6
    botMidY = cy - curve * 17 + op * 30 + 3
    return (f"M {xL} {cornerY:.1f} Q {cx} {topMidY:.1f} {xR} {cornerY:.1f} "
            f"Q {cx} {botMidY:.1f} {xL} {cornerY:.1f} Z")


def render_svg(params: dict, bg: str = "#0f1117", label: str = "") -> str:
    tilt = _g(params, "head_tilt") * 8
    pitch = _g(params, "head_pitch")
    tf = f"rotate({tilt:.2f} 210 210) translate(0 {-pitch*10:.2f}) scale({1+pitch*0.03:.4f})"
    bL, bR = _brow("L", params), _brow("R", params)
    upL, loL = _lids("L", params)
    upR, loR = _lids("R", params)
    mouth = _mouth(params)
    gx, gy = _g(params, "gaze_x") * 11, _g(params, "gaze_y") * 8
    ch = max(0.0, _g(params, "cheek_raise")) * 0.5
    pL = f'<circle cx="{150+gx:.1f}" cy="{182+gy:.1f}" r="12" fill="#2a2f3a"/>'
    pR = f'<circle cx="{270+gx:.1f}" cy="{182+gy:.1f}" r="12" fill="#2a2f3a"/>'
    text = (f'<text x="210" y="404" text-anchor="middle" fill="#8a92a6" '
            f'font-family="sans-serif" font-size="18">{label}</text>') if label else ""
    return f'''<svg viewBox="0 0 420 420" xmlns="http://www.w3.org/2000/svg">
<rect width="420" height="420" fill="{bg}" rx="16"/>
<g transform="{tf}">
  <ellipse cx="210" cy="212" rx="150" ry="172" fill="{SKIN}" stroke="{SKIN2}" stroke-width="2"/>
  <ellipse cx="150" cy="250" rx="34" ry="22" fill="#f0c6a4" opacity="{ch:.3f}"/>
  <ellipse cx="270" cy="250" rx="34" ry="22" fill="#f0c6a4" opacity="{ch:.3f}"/>
  <ellipse cx="150" cy="182" rx="34" ry="24" fill="#fff"/>
  <ellipse cx="270" cy="182" rx="34" ry="24" fill="#fff"/>
  {pL}{pR}
  <path fill="{SKIN}" d="{upL}"/><path fill="{SKIN}" d="{upR}"/>
  <path fill="{SKIN}" d="{loL}"/><path fill="{SKIN}" d="{loR}"/>
  <ellipse cx="150" cy="182" rx="34" ry="24" fill="none" stroke="{SKIN2}" stroke-width="2"/>
  <ellipse cx="270" cy="182" rx="34" ry="24" fill="none" stroke="{SKIN2}" stroke-width="2"/>
  <path stroke="#5b3b28" stroke-width="8" stroke-linecap="round" fill="none" d="{bL}"/>
  <path stroke="#5b3b28" stroke-width="8" stroke-linecap="round" fill="none" d="{bR}"/>
  <path d="M210 196 q-10 40 -16 52 q8 8 22 0" fill="none" stroke="{SKIN2}" stroke-width="2"/>
  <path fill="#b5473f" stroke="#8f3630" stroke-width="2" d="{mouth}"/>
</g>{text}
</svg>'''


def render_from_meters(meters: dict, mapper: FaceMapper | None = None, **kw) -> str:
    mapper = mapper or FaceMapper()
    return render_svg(mapper.to_params(meters), **kw)
