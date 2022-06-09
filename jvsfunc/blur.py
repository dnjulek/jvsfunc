"""
Blur functions
"""

from __future__ import annotations

import math
from typing import Sequence
from .util import ex_matrix, ex_planes
from vsutil import get_neutral_value
from vsutil import split, join
import vapoursynth as vs
core = vs.core


def gauss(src: vs.VideoNode, sigma: float = 0.5, mode: str = 'hv') -> vs.VideoNode:
    """
    The fastest gaussian blur without sigma limitation.

    :param src: Input clip.
    :param sigma: Standard deviation of gaussian blur.
    :param mode: 'h', 'v' or 'hv', to apply the filter horizontally, vertically, or both.
    """

    def gauss_conv(src: vs.VideoNode, sigma: float = 2, mode: str = 'hv') -> vs.VideoNode:

        taps = int(math.ceil(sigma * 6 + 1))
        if not taps % 2:
            taps += 1

        kernel = []
        for x in range(int(math.floor(taps / 2))):
            kernel.append(1.0 / (math.sqrt(2.0 * math.pi) * sigma) * math.exp(-(x * x) / (2 * sigma * sigma)))

        for i in range(1, len(kernel)):
            kernel[i] *= 1023 / kernel[0]
        kernel[0] = 1023

        kernel = kernel[::-1] + kernel[1:]
        return src.std.Convolution(kernel, mode=mode)

    def gauss_fmtc(src: vs.VideoNode, sigma: float = 2, mode: str = 'hv') -> vs.VideoNode:
        wsrc, hsrc = src.width, src.height

        if mode == 'h':
            wdown, hdown = round(wsrc/sigma), hsrc
        elif mode == 'v':
            wdown, hdown = wsrc, round(hsrc/sigma)
        elif mode in ['hv', 'vh']:
            wdown, hdown = round(wsrc/sigma), round(hsrc/sigma)
        else:
            raise ValueError("gauss: Invalid mode, use 'h', 'v' or 'hv'.")

        def blur(src: vs.VideoNode):
            src = src.resize.Bilinear(wdown, hdown)
            return src.fmtc.resample(wsrc, hsrc, kernel='gauss', a1=9)

        odd = wdown % 2 == 1 or hdown % 2 == 1
        if odd and src.format.color_family != vs.GRAY:
            yuv = split(src)
            yuv = [blur(i) for i in yuv]
            return join(yuv)
        else:
            return blur(src)

    if sigma < 0.334:
        return src
    elif sigma <= 4.333:
        return gauss_conv(src, sigma, mode)
    else:
        return gauss_fmtc(src, sigma, mode)


def sbr(src: vs.VideoNode, r: int = 1, mode: str = 'hv', planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    """
    A faster sbr implementation, using Convolution only once even for r 2 and 3.
    A single function for all modes (sbr, sbrH and sbrV).

    :param src: Input clip.
    :param r: 1, 2 or 3 (blur strength).
    :param mode: 'h', 'v' or 'hv', to apply the filter horizontally, vertically, or both.
    :param planes: Planes to process.
    """

    neutral = get_neutral_value(src, chroma=True)

    if mode in ['hv', 'vh']:
        matrix2 = [1, 3, 4, 3, 1]
        matrix3 = [1, 4, 8, 10, 8, 4, 1]
    elif mode in ['h', 'v']:
        matrix2 = [1, 6, 15, 20, 15, 6, 1]
        matrix3 = [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]
    else:
        raise ValueError("sbr: Invalid mode, use 'h', 'v' or 'hv'.")

    if r == 1:
        matrix = [1, 2, 1]
    elif r == 2:
        matrix = matrix2
    elif r == 3:
        matrix = matrix3
    else:
        raise ValueError("sbr: Invalid r, use 1, 2 or 3.")

    expr = [f'x y - x {neutral} - * 0 < {neutral} x y - abs x {neutral} - abs < x y - {neutral} + x ? ?']
    rg11 = src.std.Convolution(matrix=matrix, planes=planes, mode=mode)
    rg11d = core.std.MakeDiff(src, rg11, planes=planes)
    rg11ds = rg11d.std.Convolution(matrix=matrix, planes=planes, mode=mode)
    rg11dd = core.std.Expr([rg11d, rg11ds], ex_planes(rg11d, expr, planes))
    return core.std.MakeDiff(src, rg11dd, planes=planes)


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
