"""
Some functions implemented with core.akarin.Expr
"""

from vsutil import plane, depth, get_depth, split, join, scale_value, get_neutral_value
from .util import morpho_matrix, ex_matrix
from lvsfunc.scale import ssim_downsample
from lvsfunc.util import get_prop
from typing import Any
import vapoursynth as vs
core = vs.core


def ccd(src: vs.VideoNode,
        threshold: float = 4,
        mode: int = 1,
        scale: float | int = 0,
        debug: bool = False,
        matrix: int | None = None,
        **kwargs: Any) -> vs.VideoNode:
    """
    Yet another Camcorder Color Denoise implementation.

    :param src: RGBS or YUV input clip (converted to RGBS when YUV).
    :param threshold: Chroma denoise threshold.
    :param mode: How the YUV -> RGB -> YUV conversion is done:
                 mode=1: bicubic to chroma resolution, and then back to YUV with bicubic.
                 mode=2: bicubic to luma resolution, and then back to YUV with bicubic.
                 mode=3: nnedi3 to luma resolution, and then back to YUV with bicubic.
                 mode=4: nnedi3 to luma resolution, and then back to YUV with ssim_downsample.
    :param scale: Scale the matrix size for convolution, scale=1 enables the automatic resize and scale=0 disables it.
                  You can also use custom values like scale=1.5 for a matrix 50% bigger than the original.
                  The original matrix size is 25x25 for a 320x240 chroma.
                  Chroma size depends on the mode, so enable debug before using a custom value.
    :param debug: Shows the scale used (including automatic one from scale=1) and the matrix size.
    :param matrix: YUV matrix coefficient. If None, 1 will be used for HD and 6 for SD.
    :param kwargs: Arguments passed to ssim_downsample.
    """

    if src.format.color_family == vs.GRAY:
        raise ValueError('ccd: GRAY format is not supported.')

    if mode not in [1, 2, 3, 4]:
        raise ValueError('ccd: Not supported mode, use 1, 2, 3 or 4.')

    thr = threshold**2/195075.0

    def nnup(src: vs.VideoNode) -> vs.VideoNode:
        src = src.nnedi3.nnedi3(1, 1, 0, 0, 3, 2).std.Transpose()
        return src.nnedi3.nnedi3(1, 1, 0, 0, 3, 2).std.Transpose().resize.Bicubic(src_top=-0.5)

    def nnrgb(src: vs.VideoNode, matrix) -> vs.VideoNode:
        yuv = split(src)
        u, v = nnup(yuv[1]), nnup(yuv[2])
        return join([yuv[0], u, v]).resize.Point(format=vs.RGBS, matrix_in=matrix)

    def get_matrix(src: vs.VideoNode) -> int:
        frame = src.get_frame(0)
        guess = 1 if src.width > 1270 or src.height > 710 else 6

        if "_Matrix" in frame.props:
            matrix = get_prop(frame, "_Matrix", int)
        else:
            matrix = guess
        return guess if matrix == 2 else matrix

    def expr(src: vs.VideoNode, scale) -> vs.VideoNode:
        r = core.std.ShufflePlanes([src, src, src], [0, 0, 0], vs.RGB)
        g = core.std.ShufflePlanes([src, src, src], [1, 1, 1], vs.RGB)
        b = core.std.ShufflePlanes([src, src, src], [2, 2, 2], vs.RGB)

        if scale <= 0:
            x, y = 4, 12
        elif scale == 1:
            scale = src.height/240
            x = round(scale*4)
            y = round(scale*12)
        else:
            x = round(scale*4)
            y = round(scale*12)

        ex_ccd = core.akarin.Expr([r, g, b, src],
                                  f'x[-{y},-{y}] x - 2 pow y[-{y},-{y}] y - 2 pow z[-{y},-{y}] z - 2 pow + + A! '
                                  f'x[-{x},-{y}] x - 2 pow y[-{x},-{y}] y - 2 pow z[-{x},-{y}] z - 2 pow + + B! '
                                  f'x[{x},-{y}] x - 2 pow y[{x},-{y}] y - 2 pow z[{x},-{y}] z - 2 pow + + C! '
                                  f'x[{y},-{y}] x - 2 pow y[{y},-{y}] y - 2 pow z[{y},-{y}] z - 2 pow + + D! '
                                  f'x[-{y},-{x}] x - 2 pow y[-{y},-{x}] y - 2 pow z[-{y},-{x}] z - 2 pow + + E! '
                                  f'x[-{x},-{x}] x - 2 pow y[-{x},-{x}] y - 2 pow z[-{x},-{x}] z - 2 pow + + F! '
                                  f'x[{x},-{x}] x - 2 pow y[{x},-{x}] y - 2 pow z[{x},-{x}] z - 2 pow + + G! '
                                  f'x[{y},-{x}] x - 2 pow y[{y},-{x}] y - 2 pow z[{y},-{x}] z - 2 pow + + H! '
                                  f'x[-{y},{x}] x - 2 pow y[-{y},{x}] y - 2 pow z[-{y},{x}] z - 2 pow + + I! '
                                  f'x[-{x},{x}] x - 2 pow y[-{x},{x}] y - 2 pow z[-{x},{x}] z - 2 pow + + J! '
                                  f'x[{x},{x}] x - 2 pow y[{x},{x}] y - 2 pow z[{x},{x}] z - 2 pow + + K! '
                                  f'x[{y},{x}] x - 2 pow y[{y},{x}] y - 2 pow z[{y},{x}] z - 2 pow + + L! '
                                  f'x[-{y},{y}] x - 2 pow y[-{y},{y}] y - 2 pow z[-{y},{y}] z - 2 pow + + M! '
                                  f'x[-{x},{y}] x - 2 pow y[-{x},{y}] y - 2 pow z[-{x},{y}] z - 2 pow + + N! '
                                  f'x[{x},{y}] x - 2 pow y[{x},{y}] y - 2 pow z[{x},{y}] z - 2 pow + + O! '
                                  f'x[{y},{y}] x - 2 pow y[{y},{y}] y - 2 pow z[{y},{y}] z - 2 pow + + P! '
                                  f'A@ {thr} < 1 0 ? B@ {thr} < 1 0 ? C@ {thr} < 1 0 ? D@ {thr} < 1 0 ? '
                                  f'E@ {thr} < 1 0 ? F@ {thr} < 1 0 ? G@ {thr} < 1 0 ? H@ {thr} < 1 0 ? '
                                  f'I@ {thr} < 1 0 ? J@ {thr} < 1 0 ? K@ {thr} < 1 0 ? L@ {thr} < 1 0 ? '
                                  f'M@ {thr} < 1 0 ? N@ {thr} < 1 0 ? O@ {thr} < 1 0 ? P@ {thr} < 1 0 ? '
                                  '+ + + + + + + + + + + + + + + 1 + Q! '
                                  f'A@ {thr} < a[-{y},-{y}] 0 ? B@ {thr} < a[-{x},-{y}] 0 ? '
                                  f'C@ {thr} < a[{x},-{y}] 0 ? D@ {thr} < a[{y},-{y}] 0 ? '
                                  f'E@ {thr} < a[-{y},-{x}] 0 ? F@ {thr} < a[-{x},-{x}] 0 ? '
                                  f'G@ {thr} < a[{x},-{x}] 0 ? H@ {thr} < a[{y},-{x}] 0 ? '
                                  f'I@ {thr} < a[-{y},{x}] 0 ? J@ {thr} < a[-{x},{x}] 0 ? '
                                  f'K@ {thr} < a[{x},{x}] 0 ? L@ {thr} < a[{y},{x}] 0 ? '
                                  f'M@ {thr} < a[-{y},{y}] 0 ? N@ {thr} < a[-{x},{y}] 0 ? '
                                  f'O@ {thr} < a[{x},{y}] 0 ? P@ {thr} < a[{y},{y}] 0 ? '
                                  '+ + + + + + + + + + + + + + + a + Q@ /')
        if debug:
            msize = str((y*2) + 1)
            return ex_ccd.text.Text("current scale: "+str(scale)+"\nmatrix size: "+msize+"x"+msize)
        return ex_ccd

    if src.format.color_family == vs.YUV:

        if matrix is None:
            matrix = get_matrix(src)

        if src.format.subsampling_h and src.format.subsampling_w:
            src444 = src.format.replace(subsampling_w=0, subsampling_h=0).id
            cw = src.width//2
            ch = src.height//2

            if mode == 1:
                rgbs = src.resize.Bicubic(cw, ch, format=vs.RGBS, matrix_in=matrix)
                den = expr(rgbs, scale).resize.Bicubic(format=src444, matrix=matrix, src_left=-0.25)
            elif mode == 2:
                rgbs = src.resize.Bicubic(format=vs.RGBS, matrix_in=matrix)
                den = expr(rgbs, scale).resize.Bicubic(format=src.format.id, matrix=matrix)
            elif mode == 3:
                rgbs = nnrgb(src, matrix)
                den = expr(rgbs, scale).resize.Bicubic(format=src.format.id, matrix=matrix)
            else:
                rgbs = nnrgb(src, matrix)
                den = expr(rgbs, scale).resize.Bicubic(format=vs.YUV444PS, matrix=matrix, src_left=-0.5)
                u = ssim_downsample(plane(den, 1), cw, ch, **kwargs)
                v = ssim_downsample(plane(den, 2), cw, ch, **kwargs)
                den = core.std.ShufflePlanes([den, u, v], [0, 0, 0], vs.YUV)
                den = den.resize.Point(format=src.format.id)
        else:
            rgbs = src.resize.Bicubic(format=vs.RGBS, matrix_in=matrix)
            den = expr(rgbs, scale).resize.Bicubic(format=src.format.id, matrix=matrix)
        den = core.std.ShufflePlanes([src, den], [0, 1, 2], vs.YUV)
    else:
        rgbs = depth(src, 32)
        den = expr(rgbs, scale)

    if debug:
        return expr(rgbs, scale)
    return den


def ccdmod(src: vs.VideoNode, threshold: float = 4, matrix: int | None = None) -> vs.VideoNode:
    """
    A faster implementation that processes only chroma.
    """

    if src.format.color_family != vs.YUV:
        raise ValueError('ccdmod: only YUV format is supported, use jvsfunc.ccd() for RGB.')

    if matrix is None:
        matrix = 1 if src.width > 1270 or src.height > 710 else 6

    thr = threshold**2/195075.0
    u = plane(src, 1)

    yuv = src.resize.Bicubic(u.width, u.height, format=vs.YUV444P16)
    rgbs = yuv.resize.Point(format=vs.RGBS, matrix_in=matrix)

    rrr = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [0, 0, 0], vs.RGB)
    ggg = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [1, 1, 1], vs.RGB)
    bbb = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [2, 2, 2], vs.RGB)

    ex_ccd = core.akarin.Expr([yuv, rrr, ggg, bbb],
                              ['',
                               'y[-12,-12] y - 2 pow z[-12,-12] z - 2 pow a[-12,-12] a - 2 pow + + A! '
                               'y[-4,-12] y - 2 pow z[-4,-12] z - 2 pow a[-4,-12] a - 2 pow + + B! '
                               'y[4,-12] y - 2 pow z[4,-12] z - 2 pow a[4,-12] a - 2 pow + + C! '
                               'y[12,-12] y - 2 pow z[12,-12] z - 2 pow a[12,-12] a - 2 pow + + D! '
                               'y[-12,-4] y - 2 pow z[-12,-4] z - 2 pow a[-12,-4] a - 2 pow + + E! '
                               'y[-4,-4] y - 2 pow z[-4,-4] z - 2 pow a[-4,-4] a - 2 pow + + F! '
                               'y[4,-4] y - 2 pow z[4,-4] z - 2 pow a[4,-4] a - 2 pow + + G! '
                               'y[12,-4] y - 2 pow z[12,-4] z - 2 pow a[12,-4] a - 2 pow + + H! '
                               'y[-12,4] y - 2 pow z[-12,4] z - 2 pow a[-12,4] a - 2 pow + + I! '
                               'y[-4,4] y - 2 pow z[-4,4] z - 2 pow a[-4,4] a - 2 pow + + J! '
                               'y[4,4] y - 2 pow z[4,4] z - 2 pow a[4,4] a - 2 pow + + K! '
                               'y[12,4] y - 2 pow z[12,4] z - 2 pow a[12,4] a - 2 pow + + L! '
                               'y[-12,12] y - 2 pow z[-12,12] z - 2 pow a[-12,12] a - 2 pow + + M! '
                               'y[-4,12] y - 2 pow z[-4,12] z - 2 pow a[-4,12] a - 2 pow + + N! '
                               'y[4,12] y - 2 pow z[4,12] z - 2 pow a[4,12] a - 2 pow + + O! '
                               'y[12,12] y - 2 pow z[12,12] z - 2 pow a[12,12] a - 2 pow + + P! '
                               f'A@ {thr} < 1 0 ? B@ {thr} < 1 0 ? C@ {thr} < 1 0 ? D@ {thr} < 1 0 ? '
                               f'E@ {thr} < 1 0 ? F@ {thr} < 1 0 ? G@ {thr} < 1 0 ? H@ {thr} < 1 0 ? '
                               f'I@ {thr} < 1 0 ? J@ {thr} < 1 0 ? K@ {thr} < 1 0 ? L@ {thr} < 1 0 ? '
                               f'M@ {thr} < 1 0 ? N@ {thr} < 1 0 ? O@ {thr} < 1 0 ? P@ {thr} < 1 0 ? '
                               '+ + + + + + + + + + + + + + + 1 + Q! '
                               f'A@ {thr} < x[-12,-12] 0 ? B@ {thr} < x[-4,-12] 0 ? '
                               f'C@ {thr} < x[4,-12] 0 ? D@ {thr} < x[12,-12] 0 ? '
                               f'E@ {thr} < x[-12,-4] 0 ? F@ {thr} < x[-4,-4] 0 ? '
                               f'G@ {thr} < x[4,-4] 0 ? H@ {thr} < x[12,-4] 0 ? '
                               f'I@ {thr} < x[-12,4] 0 ? J@ {thr} < x[-4,4] 0 ? '
                               f'K@ {thr} < x[4,4] 0 ? L@ {thr} < x[12,4] 0 ? '
                               f'M@ {thr} < x[-12,12] 0 ? N@ {thr} < x[-4,12] 0 ? '
                               f'O@ {thr} < x[4,12] 0 ? P@ {thr} < x[12,12] 0 ? '
                               '+ + + + + + + + + + + + + + + x + Q@ /'])

    ex_ccd = ex_ccd.resize.Bicubic(src_left=-0.25)
    ex_ccd = depth(ex_ccd, get_depth(src))
    return core.std.ShufflePlanes([src, ex_ccd], [0, 1, 2], vs.YUV)


def vinverse(src: vs.VideoNode,
             sstr: float = 2.7,
             amnt: int = 255,
             chroma: bool = True,
             scl: float = 0.25) -> vs.VideoNode:
    """
    A simple filter to remove residual combing, based on an AviSynth script by Didée.

    :param src: Input clip.
    :param sstr: strength of contra sharpening
    :param amnt: change no pixel by more than this (default=255: unrestricted)
    :param chroma: chroma mode, True=process chroma, False=pass chroma through
    :param scl: scale factor for vshrpD*vblurD < 0
    """

    neutral = get_neutral_value(src)
    exp = [f'x y - {neutral} + vbd! '  # vblurD
           f'y y z - {sstr} * + round '  # vshrp
           f'y - {neutral} + vsd! '  # vshrpD
           f'vsd@ {neutral} - vbd@ {neutral} - * 0 < vsd@ {neutral} - abs '  # vlimD
           f'vbd@ {neutral} - abs < vsd@ vbd@ ? {neutral} - {scl} * {neutral} + '  # vlimD
           f'vsd@ {neutral} - abs vbd@ {neutral} - abs < vsd@ vbd@ ? ? round '  # vlimD
           f'y {neutral} - +', '']  # last

    if src.format.sample_type == vs.FLOAT:
        exp = [i.replace('round ', '') for i in exp]

    planes = [0]

    if chroma:
        planes = [0, 1, 2]
        exp = exp[0]

    vblur = src.std.Convolution(matrix=[50, 99, 50], mode='v', planes=planes)
    vblur2 = vblur.std.Convolution(matrix=[1, 4, 6, 4, 1], mode='v', planes=planes)
    vnv = core.akarin.Expr([src, vblur, vblur2], exp)

    if amnt <= 0:
        return src
    elif amnt < 255:
        AMN = scale_value(amnt, 8, get_depth(src))
        vnv = core.akarin.Expr([src, vnv], f'x {AMN} + y < x {AMN} + x {AMN} - y > x {AMN} - y ? ?')
    return vnv


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


def repair(src: vs.VideoNode, ref: vs.VideoNode, mode: int = 1) -> vs.VideoNode:
    """
    Same speed as rgvs and faster than rgsf.
    """
    mode_list = [1, 2, 3, 4, 11, 12, 13, 14]
    if mode not in mode_list:
        raise ValueError('repair: Only modes 1-4 and 11-14 are implemented.')

    pixels = 'y[-1,-1] y[0,-1] y[1,-1] y[-1,0] y[1,0] y[-1,1] y[0,1] y[1,1] '

    if mode <= 4:
        expr = f'y sort9 dup{9 - mode} max! dup{mode - 1} min! drop9 x min@ max@ clip'
    else:
        mode = mode - 10
        expr = f'sort8 dup{8 - mode} max! dup{mode - 1} min! drop8 y min@ min ymin! y max@ max ymax! x ymin@ ymax@ clip'

    return core.akarin.Expr([src, ref], pixels + expr)


def dilate(src: vs.VideoNode, size: int = 5) -> vs.VideoNode:
    """
    Same result as core.morpho.Dilate(), faster and workable in 32 bit.
    """
    expr = morpho_matrix(size, mm='max')
    return core.akarin.Expr(src, expr)


def erode(src: vs.VideoNode, size: int = 5) -> vs.VideoNode:
    """
    Same result as core.morpho.Erode(), faster and workable in 32 bit.
    """
    expr = morpho_matrix(size, mm='min')
    return core.akarin.Expr(src, expr)


def close(src: vs.VideoNode, size: int = 5) -> vs.VideoNode:
    """
    Same result as core.morpho.Close(), faster and workable in 32 bit.
    """
    close = dilate(src, size)
    return erode(close, size)


def open(src: vs.VideoNode, size: int = 5) -> vs.VideoNode:
    """
    Same result as core.morpho.Open(), faster and workable in 32 bit.
    """
    open = erode(src, size)
    return dilate(open, size)
