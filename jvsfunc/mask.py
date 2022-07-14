"""
Masks functions
"""

from __future__ import annotations

from vsutil import depth, iterate, get_depth, get_y, get_peak_value, scale_value, Dither, Range
from typing import Sequence
from .misc import retinex
from math import sqrt
import vapoursynth as vs
core = vs.core


def flat_mask(src: vs.VideoNode, radius: int = 5, thr: int = 3, use_gauss: bool = False) -> vs.VideoNode:
    """
    Binarizes the image using thresholding for a pixel based on a small region around it.

    :param src: Input clip.
    :param radius: Size of the region around the pixel to find the threshold.
    :param thr: Controls the threshold.
    :param use_gauss: Uses Gaussian blur instead of Mean/BoxBlur to filter the region around the pixel.
    """

    luma = get_y(src)

    def boxblur(src: vs.VideoNode, r: int) -> vs.VideoNode:
        if r > 12:
            return src.std.BoxBlur(0, r, 1, r, 1)
        else:
            block_size = (r * 2) + 1
            return src.std.Convolution([1] * block_size, mode='hv')

    if use_gauss:
        from vsrgtools import gauss_blur
        blur = gauss_blur(luma, radius * 0.361083333)
    else:
        blur = boxblur(luma, radius)

    mask = core.abrz.AdaptiveBinarize(depth(luma, 8), depth(blur, 8), thr)
    return depth(mask, get_depth(luma), dither_type=Dither.NONE, range_in=Range.FULL, range=Range.FULL)


def retinex_edgemask(src: vs.VideoNode,
                     tcanny_sigma: float | int = 1,
                     retinex_sigmas: Sequence[float | int] = [50, 200, 350],
                     brz: int = 8000) -> vs.VideoNode:
    """
    retinex_edgemask from kagefunc using jvsfunc.retinex rather than the plugin.

    :param src: Input clip.
    :param tcanny_sigma: sigma of tcanny.
    :param retinex_sigmas: sigma list of retinex.
    :param brz: if brz > 0 output will be binarized (brz uses 16-bit scale).
    """
    luma = get_y(src)
    max_value = get_peak_value(src)
    sbrz = scale_value(brz, 16, get_depth(src))
    ret = retinex(luma, retinex_sigmas)
    tcanny = ret.tcanny.TCanny(mode=1, sigma=tcanny_sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    kirsch1 = luma.std.Convolution(matrix=[5,  5,  5, -3,  0, -3, -3, -3, -3], saturate=False)
    kirsch2 = luma.std.Convolution(matrix=[-3,  5,  5, -3,  0,  5, -3, -3, -3], saturate=False)
    kirsch3 = luma.std.Convolution(matrix=[-3, -3,  5, -3,  0,  5, -3, -3,  5], saturate=False)
    kirsch4 = luma.std.Convolution(matrix=[-3, -3, -3, -3,  0,  5, -3,  5,  5], saturate=False)
    expr = f'x y z a max max max b + {max_value} min'
    expr_brz = f'x y z a max max max b + {sbrz} > {max_value} 0 ?'
    return core.akarin.Expr([kirsch1, kirsch2, kirsch3, kirsch4, tcanny], expr_brz if brz > 0 else expr)


def dehalo_mask(src: vs.VideoNode, expand: float = 0.5, iterations: int = 2, brz: int = 255) -> vs.VideoNode:
    """
    Based on muvsfunc.YAHRmask(), stand-alone version with some tweaks.

    :param src: Input clip. I suggest to descale (if possible) and nnedi3_rpow2 first, for a cleaner mask.
    :param expand: Expansion of edge mask.
    :param iterations: Protects parallel lines and corners that are usually damaged by YAHR.
    :param brz: Adjusts the internal line thickness.
    """
    if brz > 255 or brz < 0:
        raise ValueError('dehalo_mask: brz must be between 0 and 255.')

    src_b = depth(src, 8)
    luma = get_y(src_b)
    vEdge = core.std.Expr([luma, luma.std.Maximum().std.Maximum()], ['y x - 8 - 128 *'])
    mask1 = vEdge.tcanny.TCanny(sigma=sqrt(expand*2), mode=-1).std.Expr(['x 16 *'])
    mask2 = iterate(vEdge, core.std.Maximum, iterations)
    mask2 = iterate(mask2, core.std.Minimum, iterations)
    mask2 = mask2.std.Invert().std.Binarize(80)
    mask3 = mask2.std.Inflate().std.Inflate().std.Binarize(brz)
    mask4 = mask3 if brz < 255 else mask2
    mask4 = mask4.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    mask = core.std.Expr([mask1, mask4], ['x y min'])
    return depth(mask, get_depth(src), range=1)
