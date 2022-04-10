"""
Some functions implemented with core.akarin.Expr
"""

from .util import morpho_matrix, ex_matrix
import vapoursynth as vs
core = vs.core


def ccd(src: vs.VideoNode, threshold: float = 4) -> vs.VideoNode:
    """
    Yet another Camcorder Color Denoise implementation.
    """

    if src.format.color_family != vs.RGB or src.format.sample_type != vs.FLOAT:
        raise ValueError('ccd: only RGBS format is supported')

    thr = threshold**2/195075.0
    r = core.std.ShufflePlanes([src, src, src], [0, 0, 0], vs.RGB)
    g = core.std.ShufflePlanes([src, src, src], [1, 1, 1], vs.RGB)
    b = core.std.ShufflePlanes([src, src, src], [2, 2, 2], vs.RGB)

    ex_ccd = core.akarin.Expr([r, g, b, src],
                              'x[-12,-12] x - 2 pow y[-12,-12] y - 2 pow z[-12,-12] z - 2 pow + + A! '
                              'x[-4,-12] x - 2 pow y[-4,-12] y - 2 pow z[-4,-12] z - 2 pow + + B! '
                              'x[4,-12] x - 2 pow y[4,-12] y - 2 pow z[4,-12] z - 2 pow + + C! '
                              'x[12,-12] x - 2 pow y[12,-12] y - 2 pow z[12,-12] z - 2 pow + + D! '
                              'x[-12,-4] x - 2 pow y[-12,-4] y - 2 pow z[-12,-4] z - 2 pow + + E! '
                              'x[-4,-4] x - 2 pow y[-4,-4] y - 2 pow z[-4,-4] z - 2 pow + + F! '
                              'x[4,-4] x - 2 pow y[4,-4] y - 2 pow z[4,-4] z - 2 pow + + G! '
                              'x[12,-4] x - 2 pow y[12,-4] y - 2 pow z[12,-4] z - 2 pow + + H! '
                              'x[-12,4] x - 2 pow y[-12,4] y - 2 pow z[-12,4] z - 2 pow + + I! '
                              'x[-4,4] x - 2 pow y[-4,4] y - 2 pow z[-4,4] z - 2 pow + + J! '
                              'x[4,4] x - 2 pow y[4,4] y - 2 pow z[4,4] z - 2 pow + + K! '
                              'x[12,4] x - 2 pow y[12,4] y - 2 pow z[12,4] z - 2 pow + + L! '
                              'x[-12,12] x - 2 pow y[-12,12] y - 2 pow z[-12,12] z - 2 pow + + M! '
                              'x[-4,12] x - 2 pow y[-4,12] y - 2 pow z[-4,12] z - 2 pow + + N! '
                              'x[4,12] x - 2 pow y[4,12] y - 2 pow z[4,12] z - 2 pow + + O! '
                              'x[12,12] x - 2 pow y[12,12] y - 2 pow z[12,12] z - 2 pow + + P! '
                              f'A@ {thr} < 1 0 ? B@ {thr} < 1 0 ? C@ {thr} < 1 0 ? D@ {thr} < 1 0 ? '
                              f'E@ {thr} < 1 0 ? F@ {thr} < 1 0 ? G@ {thr} < 1 0 ? H@ {thr} < 1 0 ? '
                              f'I@ {thr} < 1 0 ? J@ {thr} < 1 0 ? K@ {thr} < 1 0 ? L@ {thr} < 1 0 ? '
                              f'M@ {thr} < 1 0 ? N@ {thr} < 1 0 ? O@ {thr} < 1 0 ? P@ {thr} < 1 0 ? '
                              '+ + + + + + + + + + + + + + + 1 + Q! '
                              f'A@ {thr} < a[-12,-12] 0 ? B@ {thr} < a[-4,-12] 0 ? '
                              f'C@ {thr} < a[4,-12] 0 ? D@ {thr} < a[12,-12] 0 ? '
                              f'E@ {thr} < a[-12,-4] 0 ? F@ {thr} < a[-4,-4] 0 ? '
                              f'G@ {thr} < a[4,-4] 0 ? H@ {thr} < a[12,-4] 0 ? '
                              f'I@ {thr} < a[-12,4] 0 ? J@ {thr} < a[-4,4] 0 ? '
                              f'K@ {thr} < a[4,4] 0 ? L@ {thr} < a[12,4] 0 ? '
                              f'M@ {thr} < a[-12,12] 0 ? N@ {thr} < a[-4,12] 0 ? '
                              f'O@ {thr} < a[4,12] 0 ? P@ {thr} < a[12,12] 0 ? '
                              '+ + + + + + + + + + + + + + + a + Q@ /')

    return ex_ccd


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