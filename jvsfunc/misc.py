"""
Miscellaneous functions
"""

from __future__ import annotations

from functools import partial
from typing import List
from vsutil import get_depth, get_y, scale_value, EXPR_VARS
from .util import _morpho_matrix
import vapoursynth as vs
core = vs.core


def retinex(src: vs.VideoNode,
            sigmas: List[float | int] = [25, 80, 250],
            lower_thr: float = 0.001,
            upper_thr: float = 0.001,
            fast: bool = False
            ) -> vs.VideoNode:
    """
    A Multi Scale Retinex implementation that can be faster than the current VS plugin.

    :param src: Input clip.
    :param sigmas: Sigma list for Gaussian blur.
    :param lower_thr: Controls the white balance.
    :param upper_thr: Controls the white balance.
    :param fast: Replaces the strongest Gaussian blur with PlaneStatsMax.
    """

    from vsrgtools import gauss_blur

    luma = get_y(src).std.PlaneStats()
    is_float = luma.format.sample_type == vs.FLOAT

    if is_float:
        luma_float = luma.akarin.Expr("x x.PlaneStatsMin - x.PlaneStatsMax x.PlaneStatsMin - /")
    else:
        luma_float = luma.akarin.Expr("1 x.PlaneStatsMax x.PlaneStatsMin - / x x.PlaneStatsMin - *", format=vs.GRAYS)

    slen = len(sigmas)
    slen_fast = (slen - 1) if fast else slen
    ev_list = [EXPR_VARS[i+1] for i in range(slen_fast)]
    expr_msr = "".join([f"{i} 0 <= 1 x {i} / 1 + ? " for i in ev_list])

    if fast:
        expr_msr = expr_msr + "x.PlaneStatsMax 0 <= 1 x x.PlaneStatsMax / 1 + ? "
        sigmas.sort()
        sigmas = sigmas[:-1]

    expr_msr = expr_msr + f"{'+ ' * (slen-1)}log {slen} /"
    blur_list = [gauss_blur(luma_float, i) for i in sigmas]
    msr = core.akarin.Expr([luma_float] + blur_list, expr_msr)
    msr_stats = msr.psm.PlaneMinMax(lower_thr, upper_thr)
    expr_balance = "x x.psmMin - x.psmMax x.psmMin - /"

    if not is_float:
        _floor = scale_value(16, 8, get_depth(luma))
        _ceil = scale_value(235, 8, get_depth(luma))
        _range = _ceil - _floor
        expr_balance = expr_balance + f" {_range} * {_floor} + round {_floor} {_ceil} clamp"

    return msr_stats.akarin.Expr(expr_balance, format=luma.format)


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
    expr = _morpho_matrix(size, mm='max')
    return core.akarin.Expr(src, expr)


def erode(src: vs.VideoNode, size: int = 5) -> vs.VideoNode:
    """
    Same result as core.morpho.Erode(), faster and workable in 32 bit.
    """
    expr = _morpho_matrix(size, mm='min')
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


def FreezeFramesMod(src: vs.VideoNode, mode: str = 'prev', ranges: List[int] = [],
                    single: bool = False) -> vs.VideoNode:
    """
    Mod to easily replace frames with the previous or next one.

    :param src: Input clip.
    :param mode: Frame to be copied, 1 or 'prev' for previous one, 2 or 'next' for next one.
    :param ranges: List of frame ranges to be processed.
    :param single: If True, each frame in the list will be processed individually (like a keyframe list),
                   if False, it will be taken as ranges.
    """
    frame_a = ranges if single else ranges[::2]
    frame_b = ranges if single else ranges[1::2]

    if mode in [1, 'prev']:
        replace = src.std.FreezeFrames(frame_a, frame_b, [i-1 for i in frame_a])
    elif mode in [2, 'next']:
        replace = src.std.FreezeFrames(frame_a, frame_b, [i+1 for i in frame_b])
    else:
        raise ValueError('FreezeFramesMod: unknown mode')
    return replace


ffmod = FreezeFramesMod


def replace_keyframe(src: vs.VideoNode, thr: float = 0.30, kf_list: List[int] | None = None,
                     show_thr: bool = False) -> vs.VideoNode:
    """
    Replace the frame after a scene change with the next frame. Helps to fix broken keyframes.

    :param src: Input clip.
    :param thr: Threshold is the difference between the two frames, it must be high enough to ignore artifacts
                and low enough to avoid freezing moving scenes.
    :param kf_list: List of keyframes, can be made with lvsfunc.render.find_scene_changes(clip, scxvid=True),
                    if it is not provided, scxvid will generate the keyframes in real time,
                    which can be bad for the preview.
    :param show_thr: Shows the threshold of the frame. This also disable the replacement
                     for a better frame seeking in preview.
    """
    def _show_thr(n: int, f: vs.VideoFrame, clip: vs.VideoNode):
        diff = str(f.props['PlaneStatsDiff'] * 100)  # type:ignore
        return clip.sub.Subtitle("Frame " + str(n) + " of " + str(clip.num_frames) + f"\nCurrent thr: {diff}")

    def _schang_xvid(n: int, f: vs.VideoFrame, clip: vs.VideoNode, thr: float):
        if f.props['_SceneChangePrev'] == 1 and f.props['PlaneStatsDiff'] * 100 < thr:  # type:ignore
            return clip[n+1]
        else:
            return clip

    def _schang_list(n: int, f: vs.VideoFrame, clip: vs.VideoNode, thr: float, klist: List[int]):
        if n in klist and f.props['PlaneStatsDiff'] * 100 < thr:  # type:ignore
            return clip[n+1]
        else:
            return clip

    diff = core.std.PlaneStats(src, src[1:])

    if show_thr:
        return core.std.FrameEval(src, partial(_show_thr, clip=src), prop_src=diff)

    if kf_list is None:
        xvid = diff.resize.Bilinear(640, 360, format=vs.YUV420P8).scxvid.Scxvid()
        return core.std.FrameEval(src, partial(_schang_xvid, clip=src, thr=thr), prop_src=xvid)

    return core.std.FrameEval(src, partial(_schang_list, clip=src, thr=thr, klist=kf_list), prop_src=diff)
