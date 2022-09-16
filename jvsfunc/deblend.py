"""
Deblend functions
"""

from __future__ import annotations

from functools import partial
from typing import List, Sequence
from vsutil import scale_value, get_depth
from .util import _ex_planes
from .blur import sbr
import vapoursynth as vs
core = vs.core


def _inter_pattern(clipa: List[vs.VideoNode], clipb: List[vs.VideoNode]) -> List[vs.VideoNode]:
    inter0 = core.std.Interleave([clipb[0], clipa[1], clipa[2], clipa[3], clipa[4]])
    inter1 = core.std.Interleave([clipa[0], clipb[1], clipa[2], clipa[3], clipa[4]])
    inter2 = core.std.Interleave([clipa[0], clipa[1], clipb[2], clipa[3], clipa[4]])
    inter3 = core.std.Interleave([clipa[0], clipa[1], clipa[2], clipb[3], clipa[4]])
    inter4 = core.std.Interleave([clipa[0], clipa[1], clipa[2], clipa[3], clipb[4]])
    return [inter0, inter1, inter2, inter3, inter4]


def _jdeblend_eval(n: int, f: List[vs.VideoFrame], src: vs.VideoNode, inters: List[vs.VideoNode]) -> vs.VideoNode:
    comb = [f[i].props['_Combed'] for i in [0, 1]]
    pattern = n % 5
    if comb[0] == 1:
        src = inters[pattern]
    return src[n+1] if sum(comb) == 2 else src  # type:ignore


def jdeblend(src_fm: vs.VideoNode, src: vs.VideoNode, vnv: bool = True) -> vs.VideoNode:
    """
    Automatically deblends if normal field matching leaves 2 blends every 5 frames (like avs's ExBlend).

    :param src_fm: Source after field matching, must have field=3 and low cthresh.
    :param src: Untouched source.
    :param vnv: Enable vinverse for post processing.

    Example:
    src = clip
    vfm = src.vivtc.VFM(order=1, field=3, cthresh=3)
    dblend = jdeblend(vfm, src)
    dblend = core.std.ShufflePlanes([vfm, dblend], [0, 1, 2], vs.YUV)
    dblend = jdeblend_kf(dblend, vfm)
    """

    a, ab, bc, c = src[0] + src[:-1], src, src[1:] + src[-1], src[2:] + src[-2:]
    dbd = core.std.Expr([a, ab, bc, c], "z a 2 / - y x 2 / - +")
    dbd = vinverse(dbd) if vnv else dbd

    select_src = [src.std.SelectEvery(5, i) for i in range(5)]
    select_dbd = [dbd.std.SelectEvery(5, i) for i in range(5)]
    inters = _inter_pattern(select_src, select_dbd)
    psrc = [src_fm, src_fm[0]+src_fm[:-1]]
    return core.std.FrameEval(src_fm, partial(_jdeblend_eval, src=src_fm, inters=inters), psrc)


def jdeblend_bob(src_fm: vs.VideoNode, bobbed: vs.VideoNode) -> vs.VideoNode:
    """
    Stronger version of jdeblend() that uses a bobbed clip to deblend.

    :param src_fm: Source after field matching, must have field=3 and low cthresh.
    :param bobbed: Bobbed source.

    Example:
    src = clip
    from havsfunc import QTGMC
    qtg = QTGMC(src, TFF=True, SourceMatch=3)
    vfm = src.vivtc.VFM(order=1, field=3, cthresh=3)
    dblend = jdeblend_bob(vfm, qtg)
    dblend = jdeblend_kf(dblend, vfm)
    """

    bob0 = bobbed.std.SelectEvery(2, 0)
    bob1 = bobbed.std.SelectEvery(2, 1)
    ab0, bc0, c0 = bob0, bob0[1:] + bob0[-1], bob0[2:] + bob0[-2]
    a1, ab1, bc1 = bob1[0] + bob1[:-1], bob1, bob1[1:] + bob1[-1]
    dbd = core.std.Expr([a1, ab1, ab0, bc1, bc0, c0], 'y x - z + b c - a + + 2 /')
    dbd = core.std.ShufflePlanes([bc0, dbd], [0, 1, 2], vs.YUV)

    select_src = [src_fm.std.SelectEvery(5, i) for i in range(5)]
    select_dbd = [dbd.std.SelectEvery(5, i) for i in range(5)]
    inters = _inter_pattern(select_src, select_dbd)
    psrc = [src_fm, src_fm[0]+src_fm[:-1]]
    return core.std.FrameEval(src_fm, partial(_jdeblend_eval, src=src_fm, inters=inters), psrc)


def jdeblend_kf(src: vs.VideoNode, src_fm: vs.VideoNode) -> vs.VideoNode:
    """
    Should be used after jdeblend() to fix scene changes.

    :param src: Source after jdeblend().
    :param src_fm: Source after field matching, must have field=3 and low cthresh.
    """

    def keyframe(n: int, f: List[vs.VideoFrame], src: vs.VideoNode):
        keyfm = [f[i].props['VFMSceneChange'] for i in [0, 1]]
        kf_end = keyfm[0] > keyfm[1]  # type:ignore
        kf_start = sum(keyfm) == 2  # type:ignore
        is_cmb = f[0].props['_Combed'] == 1
        src = src[n-1] if kf_end and is_cmb else src
        return src[n+1] if kf_start and is_cmb else src

    psrc = [src_fm, src_fm[0]+src_fm[:-1]]
    return core.std.FrameEval(src, partial(keyframe, src=src), psrc)


def JIVTC_Deblend(src: vs.VideoNode, pattern: int, chroma_only: bool = True, tff: bool = True) -> vs.VideoNode:
    """
    fvsfunc.JIVTC() modified to use a deblend based on lvsfunc instead of original bobber (yadifmod).

    JIVTC_Deblend works similar to the original, and follows the same A, AB, BC, C, D pattern.
    This function should only be used when a normal ivtc or ivtc+bobber leaves chroma blend to a every fourth frame.
    You can disable chroma_only to use in luma as well, but it is not recommended.

    :param src: Source clip. Has to be 60i (30000/1001).
    :param pattern: First frame of any clean-combed-combed-clean-clean sequence.
    :param chroma_only: If set to False, luma will also receive deblend process.
    :param tff: Set top field first (True) or bottom field first (False).
    """

    if src.fps_num != 30000 or src.fps_den != 1001:
        raise ValueError('JIVTC_Deblend: This filter can only be used with 60i clips.')

    pattern = pattern % 5
    cycle10 = [[0, 3, 6, 8], [0, 2, 5, 8], [0, 2, 4, 7], [2, 4, 6, 9], [1, 4, 6, 8]]
    cycle08 = [[0, 3, 4, 6], [0, 2, 5, 6], [0, 2, 4, 7], [0, 2, 4, 7], [1, 2, 4, 6]]
    cycle05 = [[0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4]]
    ivtced = core.std.SeparateFields(src, tff=tff).std.DoubleWeave()
    ivtced = core.std.SelectEvery(ivtced, 10, cycle10[pattern])

    a, ab, bc, c = src[0] + src[:-1], src, src[1:] + src[-1], src[2:] + src[-2:]
    deblended = core.std.Expr([a, ab, bc, c], "z a 2 / - y x 2 / - +")
    deblended = vinverse(deblended)
    deblended = core.std.SelectEvery(deblended, 5, cycle05[pattern])

    inter = core.std.Interleave([ivtced, deblended])
    final = core.std.SelectEvery(inter, 8, cycle08[pattern])

    final_y = core.std.ShufflePlanes([ivtced, final], [0, 1, 2], vs.YUV)
    final = final_y if chroma_only else final
    return core.std.SetFrameProp(final, prop='_FieldBased', intval=0)


def vinverse(src: vs.VideoNode,
             sstr: float = 2.7,
             amnt: int = 255,
             scl: float = 0.25,
             mode: str = 'v',
             planes: int | Sequence[int] | None = None,
             vinverse2: bool = False) -> vs.VideoNode:
    """
    A simple filter to remove residual combing or dot crawl when vinverse2=True, based on an AviSynth script by Didée.

    :param src: Input clip.
    :param sstr: strength of contra sharpening.
    :param amnt: change no pixel by more than this (default=255: unrestricted).
    :param scl: scale factor for vshrpD*vblurD < 0.
    :param mode: 'h', 'v' or 'hv', to apply the filter horizontally, vertically, or both.
    :param planes: Planes to process.
    :param vinverse2: Use Vinverse2 mode.
    """

    expr = f'y z - {sstr} * D1! x y - D2! D1@ abs D1A! D2@ abs D2A! '
    expr += f'D1@ D2@ xor D1A@ D2A@ < D1@ D2@ ? {scl} * D1A@ D2A@ < D1@ D2@ ? ? y + '

    if vinverse2:
        blur = sbr(src, mode=mode, planes=planes)
        blur2 = blur.std.Convolution([1, 2, 1], mode=mode, planes=planes)
    else:
        blur = src.std.Convolution([50, 99, 50], mode=mode, planes=planes)
        blur2 = blur.std.Convolution([1, 4, 6, 4, 1], mode=mode, planes=planes)

    if amnt <= 0:
        return src
    elif amnt < 255:
        amn = scale_value(amnt, 8, get_depth(src))
        expr += f'LIM! x {amn} + LIM@ < x {amn} + x {amn} - LIM@ > x {amn} - LIM@ ? ?'

    vnv = core.akarin.Expr([src, blur, blur2], _ex_planes(src, [expr], planes))
    return vnv
