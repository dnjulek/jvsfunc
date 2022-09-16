"""
Masks functions
"""

from __future__ import annotations
from typing import List, Sequence

from vsutil import depth, iterate, get_depth, get_y, get_peak_value, scale_value, Dither, Range
from .util import _ex_planes
from .misc import retinex
from math import sqrt
import vapoursynth as vs
core = vs.core


def clahe_edgemask(src: vs.VideoNode,
                   tcanny_sigma: float | int = 1,
                   clahe_limit: float | int = 2000,
                   clahe_tile: int = 5,
                   brz: int = 8000) -> vs.VideoNode:
    """
    Like retinex_edgemask, but using CLAHE.

    :param src: Input clip.
    :param tcanny_sigma: sigma of tcanny.
    :param clahe_limit: Threshold for contrast limiting.
    :param clahe_tile: Tile size for histogram equalization.
    :param brz: if brz > 0 output will be binarized (brz uses 16-bit scale).
    """
    luma16 = depth(get_y(src), 16)
    clahe = luma16.ehist.CLAHE(clahe_limit, clahe_tile)
    tcanny = clahe.tcanny.TCanny(mode=1, sigma=tcanny_sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    kirsch1 = luma16.std.Convolution(matrix=[5,  5,  5, -3,  0, -3, -3, -3, -3], saturate=False)
    kirsch2 = luma16.std.Convolution(matrix=[-3,  5,  5, -3,  0,  5, -3, -3, -3], saturate=False)
    kirsch3 = luma16.std.Convolution(matrix=[-3, -3,  5, -3,  0,  5, -3, -3,  5], saturate=False)
    kirsch4 = luma16.std.Convolution(matrix=[-3, -3, -3, -3,  0,  5, -3,  5,  5], saturate=False)
    expr = 'x y z a max max max b + 65535 min'
    expr_brz = f'x y z a max max max b + {brz} > 65535 0 ?'
    mask = core.akarin.Expr([kirsch1, kirsch2, kirsch3, kirsch4, tcanny], expr_brz if brz > 0 else expr)
    return depth(mask, get_depth(src), dither_type=Dither.NONE, range_in=Range.FULL, range=Range.FULL)


def comb_mask(src: vs.VideoNode,
              cthresh: int = 6,
              mthresh: int = 9,
              expand: bool = True,
              metric: int = 0,
              planes: int | Sequence[int] | None = None
              ) -> vs.VideoNode:
    """
    Comb mask from TIVTC/TFM plugin.

    :param src: Input clip.
    :param cthresh: Spatial combing threshold.
    :param mthresh: Motion adaptive threshold.
    :param expand: Assume left and right pixels of combed pixel as combed too.
    :param metric: Sets which spatial combing metric is used to detect combed pixels.
                   Metric 0 is what TFM used previous to v0.9.12.0.
                   Metric 1 is from Donald Graft's decomb.dll.
    :param planes: Planes to process.
    """
    cth_max = 65025 if metric else 255
    if (cthresh > cth_max) or (cthresh < 0):
        raise ValueError(f'comb_mask: cthresh must be between 0 and {cth_max} when metric = {metric}.')
    if (mthresh > 255) or (mthresh < 0):
        raise ValueError('comb_mask: mthresh must be between 0 and 255.')

    peak = get_peak_value(src)
    ex_m0 = [f'x[0,-2] a! x[0,-1] b! x c! x[0,1] d! x[0,2] e! '
             f'c@ b@ - d1! c@ d@ - d2! '
             f'c@ 4 * a@ + e@ + b@ d@ + 3 * - abs fd! '
             f'd1@ {cthresh} > d2@ {cthresh} > and '
             f'd1@ -{cthresh} < d2@ -{cthresh} < and or '
             f'fd@ {cthresh * 6} > and {peak} 0 ?']

    ex_m1 = [f'x[0,-1] x - x[0,1] x - * {cthresh} > {peak} 0 ?']
    ex_motion = [f'x y - abs {mthresh} > {peak} 0 ?']
    ex_spatial = ex_m1 if metric else ex_m0

    spatial_mask = src.akarin.Expr(_ex_planes(src, ex_spatial, planes))
    if (mthresh == 0):
        if not expand:
            return spatial_mask
        return spatial_mask.std.Maximum(planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])

    motion_mask = core.akarin.Expr([src, src[0] + src], _ex_planes(src, ex_motion, planes))
    motion_mask = motion_mask.std.Maximum(planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
    comb_mask = core.akarin.Expr([spatial_mask, motion_mask], 'x y min')
    if not expand:
        return comb_mask

    return comb_mask.std.Maximum(planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])


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
                     retinex_sigmas: List[float | int] = [50, 200, 350],
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
    ret = retinex(luma, retinex_sigmas, 0.001, 0.005)
    tcanny = ret.tcanny.TCanny(mode=1, sigma=tcanny_sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    kirsch1 = luma.std.Convolution(matrix=[5,  5,  5, -3,  0, -3, -3, -3, -3], saturate=False)
    kirsch2 = luma.std.Convolution(matrix=[-3,  5,  5, -3,  0,  5, -3, -3, -3], saturate=False)
    kirsch3 = luma.std.Convolution(matrix=[-3, -3,  5, -3,  0,  5, -3, -3,  5], saturate=False)
    kirsch4 = luma.std.Convolution(matrix=[-3, -3, -3, -3,  0,  5, -3,  5,  5], saturate=False)
    expr = f'x y z a max max max b + {max_value} min'
    expr_brz = f'x y z a max max max b + {sbrz} > {max_value} 0 ?'
    return core.akarin.Expr([kirsch1, kirsch2, kirsch3, kirsch4, tcanny], expr_brz if brz > 0 else expr)


def dehalo_mask(src: vs.VideoNode, expand: float = 0.5, iterations: int = 2, brz: int = 255, shift: int = 8) -> vs.VideoNode:
    """
    Based on muvsfunc.YAHRmask(), stand-alone version with some tweaks.

    :param src: Input clip. I suggest to descale (if possible) and nnedi3_rpow2 first, for a cleaner mask.
    :param expand: Expansion of edge mask.
    :param iterations: Protects parallel lines and corners that are usually damaged by YAHR.
    :param brz: Adjusts the internal line thickness.
    :param shift: Corrective shift for fine-tuning iterations
    """
    if brz > 255 or brz < 0:
        raise ValueError('dehalo_mask: brz must be between 0 and 255.')

    src_b = depth(src, 8)
    luma = get_y(src_b)
    vEdge = core.std.Expr([luma, luma.std.Maximum().std.Maximum()], [f'y x - {shift} - 128 *'])
    mask1 = vEdge.tcanny.TCanny(sigma=sqrt(expand*2), mode=-1).std.Expr(['x 16 *'])
    mask2 = iterate(vEdge, core.std.Maximum, iterations)
    mask2 = iterate(mask2, core.std.Minimum, iterations)
    mask2 = mask2.std.Invert().std.Binarize(80)
    mask3 = mask2.std.Inflate().std.Inflate().std.Binarize(brz)
    mask4 = mask3 if brz < 255 else mask2
    mask4 = mask4.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    mask = core.std.Expr([mask1, mask4], ['x y min'])
    return depth(mask, get_depth(src), range=1)
