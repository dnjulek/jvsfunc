"""
Blur functions
"""

from .util import ex_matrix
import vapoursynth as vs
core = vs.core


def medianblur(src: vs.VideoNode, radius: int = 2) -> vs.VideoNode:
    """
    A spatial median blur filter with a variable radius.
    """
    rb = radius * 2 + 1
    rb = rb * rb
    st = rb - 1
    sp = rb//2 - 1
    dp = st - 2
    expr = f'sort{st} swap{sp} min! swap{sp} max! drop{dp} x min@ max@ clip'
    expr = ex_matrix(radius) + expr
    return src.akarin.Expr(expr)
