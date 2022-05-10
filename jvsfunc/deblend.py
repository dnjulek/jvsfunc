"""
Deblend functions
"""

from functools import partial
from typing import List
from .util import inter_pattern, jdeblend_eval
from .expr import vinverse as ex_vinverse
import vapoursynth as vs
core = vs.core


def jdeblend(src_fm: vs.VideoNode, src: vs.VideoNode, vinverse: bool = True) -> vs.VideoNode:
    """
    Automatically deblends if normal field matching leaves 2 blends every 5 frames (like avs's ExBlend).

    :param src_fm: Source after field matching, must have field=3 and low cthresh.
    :param src: Untouched source.
    :param vinverse: Vinverse for post processing.

    Example:
    src = clip
    vfm = src.vivtc.VFM(order=1, field=3, cthresh=3)
    dblend = jdeblend(vfm, src)
    dblend = core.std.ShufflePlanes([vfm, dblend], [0, 1, 2], vs.YUV)
    dblend = jdeblend_kf(dblend, vfm)
    """

    a, ab, bc, c = src[0] + src[:-1], src, src[1:] + src[-1], src[2:] + src[-2:]
    dbd = core.std.Expr([a, ab, bc, c], "z a 2 / - y x 2 / - +")
    dbd = ex_vinverse(dbd) if vinverse else dbd

    select_src = [src.std.SelectEvery(5, i) for i in range(5)]
    select_dbd = [dbd.std.SelectEvery(5, i) for i in range(5)]
    inters = inter_pattern(select_src, select_dbd)
    psrc = [src_fm, src_fm[0]+src_fm[:-1]]
    return core.std.FrameEval(src_fm, partial(jdeblend_eval, src=src_fm, inters=inters), psrc)


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
    inters = inter_pattern(select_src, select_dbd)
    psrc = [src_fm, src_fm[0]+src_fm[:-1]]
    return core.std.FrameEval(src_fm, partial(jdeblend_eval, src=src_fm, inters=inters), psrc)


def jdeblend_kf(src: vs.VideoNode, src_fm: vs.VideoNode) -> vs.VideoNode:
    """
    Should be used after jdeblend() to fix scene changes.

    :param src: Source after jdeblend().
    :param src_fm: Source after field matching, must have field=3 and low cthresh.
    """

    def keyframe(n: int, f: List[vs.VideoFrame], src: vs.VideoNode):
        keyfm = [f[i].props['VFMSceneChange'] for i in [0, 1]]
        kf_end = keyfm[0] > keyfm[1]  # type:ignore
        kf_start = sum(keyfm) == 2
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
    deblended = ex_vinverse(deblended)
    deblended = core.std.SelectEvery(deblended, 5, cycle05[pattern])

    inter = core.std.Interleave([ivtced, deblended])
    final = core.std.SelectEvery(inter, 8, cycle08[pattern])

    final_y = core.std.ShufflePlanes([ivtced, final], [0, 1, 2], vs.YUV)
    final = final_y if chroma_only else final
    return core.std.SetFrameProp(final, prop='_FieldBased', intval=0)
