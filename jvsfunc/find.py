"""
Some functions to generate VSEdit Bookmarks file with a list of frames, to be used in scene filtering.
"""

from lvsfunc.render import clip_async_render
from .util import bookmarks, rng
from vsutil import depth
import operator as opr
from typing import Any
import vapoursynth as vs
core = vs.core


def find_prop(src: vs.VideoNode,
              prop: str,
              operator: str,
              ref: int | float,
              name: str = "find_prop",
              return_ranges: bool = False,
              min_length: int = 24,
              ) -> vs.VideoNode:
    """
    Creates a VSEdit Bookmarks file with a list of frames, based on the given ``prop``, ``operator`` and ``ref``.
    Check find_comb, find_30p or find_60p for examples.

    :param src: Input clip.
    :param prop: Frame prop to be used.
    :param operator: Conditional operator to apply between prop and ref ("<", "<=", "==", "!=", ">" or ">=").
    :param ref: Value to be compared with prop.
    :param name: Output file name.
    :param return_ranges: Return only the first and last frame of each sequence.
    :param min_length: Amount of frames to finish a sequence, to avoid false negatives.
    """

    ops = {
        "<": opr.lt,
        "<=": opr.le,
        "==": opr.eq,
        "!=": opr.ne,
        ">": opr.gt,
        ">=": opr.ge,
    }

    frames = []
    dbug = "Searching "+prop+" "+operator+" "+str(ref)+"..."

    def _cb(n: int, f: vs.VideoFrame):
        fprop = f.props[prop]
        if ops[operator](fprop, ref):  # type: ignore
            frames.append(n)

    clip_async_render(src, progress=dbug, callback=_cb)
    if return_ranges:
        frames = rng(frames, min_length)
    frames = bookmarks(frames, name)
    return src


def find_comb(src: vs.VideoNode, name: str = 'comb_list', **kwargs: Any) -> vs.VideoNode:
    """
    Creates a VSEdit Bookmarks file with a list of combed frames, to be used in scene filtering.

    :param src: Input clip.
    :param name: Output file name.
    :param kwargs: Arguments passed to tdm.IsCombed.
    """
    src = depth(src, 8)
    find = core.tdm.IsCombed(src, **kwargs)
    return find_prop(find, prop="_Combed", operator="==", ref=1, name=name)


def find_30p(src: vs.VideoNode,
             min_length: int = 34,
             thr: int = 2000,
             name: str = '30p_ranges',
             show_thr: bool = False
             ) -> vs.VideoNode:
    """
    Creates a VSEdit Bookmarks file with possible 30 fps ranges from a VFR (interlaced) clip,
    to be used in scene filtering.

    :param src: Input clip. Has to be 60i (30000/1001).
    :param min_length: Non 30 fps consecutive frames needed to end a range.
    :param thr: Threshold for the frame not to be considered a duplicate.
    :param name: Output file name.
    :param show_thr: Shows the frame threshold.
    """
    if src.fps_num != 30000 or src.fps_den != 1001:
        raise ValueError('find_30p: This script can only be used with 60i clips.')

    find = core.vivtc.VFM(src, order=1, mode=0).vivtc.VDecimate(dryrun=True)
    if show_thr:
        return core.text.FrameProps(find, "VDecimateMaxBlockDiff")

    find = find_prop(find, prop="VDecimateMaxBlockDiff", operator=">", ref=thr,
                     name=name, return_ranges=True, min_length=min_length)
    return find


def find_60p(src: vs.VideoNode, min_length: int = 60, name: str = '60p_ranges') -> vs.VideoNode:
    """
    Creates a VSEdit Bookmarks file with possible 60 fps ranges from a VFR (interlaced) clip,
    to be used in scene filtering.

    :param src: Input clip. Has to be 60i (30000/1001).
    :param min_length: Non 60 fps consecutive frames needed to end a range.
    :param name: Output file name.
    """
    if src.fps_num != 30000 or src.fps_den != 1001:
        raise ValueError('find_60p: This script can only be used with 60i clips.')

    find = core.vivtc.VFM(src, order=1, mode=2)
    find = find_prop(find, prop="_Combed", operator="==", ref=0, name=name,
                     return_ranges=True, min_length=min_length)
    return find
