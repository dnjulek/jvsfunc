"""
Functions to remove dotcrawl
"""

from __future__ import annotations

from vsutil import get_depth, get_y, depth, get_neutral_value, scale_value
from .deblend import vinverse
from .misc import repair
from .blur import sbr
from typing import Any, Dict
import vapoursynth as vs
core = vs.core


def ddcomb(src: vs.VideoNode, **frfun7over: Any) -> vs.VideoNode:
    """
    Based on an AviSynth script by real.finder:
    https://github.com/realfinder/AVS-Stuff/blob/Community/avs%202.5%20and%20up/DDComb.avsi
    And originally written by Didée:
    https://forum.doom9.org/showthread.php?p=1584186#post1584186

    :param src: Input clip.
    :param frfun7over: Frfun7 parameter overrides.
    """

    from lvsfunc.util import padder
    frfun7args: Dict[str, Any] = dict(l=1.01, t=8.0, p=0, tp1=0, r1=3)
    frfun7args.update(frfun7over)

    luma = get_y(src)
    v1 = scale_value(1, 8, get_depth(src))
    v2 = v1 + v1
    v3 = v2 + v1
    n = get_neutral_value(src, chroma=True)
    luma = padder(luma, left=4, right=4, top=0, bottom=0)
    w, h = luma.width, luma.height

    clean1 = luma.std.SeparateFields(True)
    clean1 = vinverse(clean1, vinverse2=True, mode='v').std.DoubleWeave(True).std.SelectEvery(2, 0)
    d1 = core.std.MakeDiff(luma, clean1)
    d8 = d1.resize.Bicubic(w//2-72, h, filter_param_a=1/3, filter_param_b=1/3)
    d8 = d8.std.Convolution([1, 1, 1], mode='h')
    d8 = d8.resize.Bicubic(w, h, filter_param_a=1, filter_param_b=0)
    d9 = core.std.Expr([d1, d8], f'x {n} - y {n} - * 0 < {n} x {n} - abs y {n} - abs < x y ? ?')
    clean1a = core.std.MergeDiff(clean1, d9)
    clean1b = depth(clean1a, 8).frfun7.Frfun7(tuv=0, **frfun7args)
    clean1b = depth(clean1b, get_depth(src))
    allD = core.std.MakeDiff(luma, clean1b)
    shrpD = core.std.MakeDiff(clean1b, clean1b.std.Convolution([1]*9))
    dd = repair(shrpD, allD, 13)
    dd = core.std.Expr([dd, shrpD], f'x {n} - y {n} - * 0 < {n} x {n} - abs y {n} - abs < x y ? ?')
    expr = f'x {v3} + y < x {v2} + x y < x {v1} + x {v3} - y > x {v2} - x y > x {v1} - x ? ? ? ?'
    clean1c = core.std.Expr([clean1b, clean1a], expr)
    clean1c = core.std.MergeDiff(clean1c, sbr(dd)).std.Crop(4, 4)
    return core.std.ShufflePlanes([clean1c, src], [0, 1, 2], vs.YUV)
