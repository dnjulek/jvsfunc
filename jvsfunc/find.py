"""
Some functions implemented with core.akarin.Expr
"""

from lvsfunc.render import clip_async_render
from lvsfunc.util import get_prop
from .util import bookmarks, rng
from vsutil import depth
import vapoursynth as vs
core = vs.core


def find_comb(src: vs.VideoNode, name: str = 'comb_list') -> vs.VideoNode:
    """
    Creates a VSEdit Bookmarks file with a list of combed frames, to be used in scene filtering.

    :param src: Input clip.
    :param name: Output file name.
    """
    frames = []
    src = depth(src, 8)
    find = core.tdm.IsCombed(src)

    def _cb(n, f):
        if get_prop(f, "_Combed", int) == 1:
            frames.append(n)

    clip_async_render(find, progress="Searching combed...", callback=_cb)
    frames = bookmarks(frames, name)
    return find


def find_30p(src: vs.VideoNode, min_length: int = 34, thr: int = 2000, name: str = '30p_ranges',
             show_thr: bool = False) -> vs.VideoNode:
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

    frames = []
    find = core.vivtc.VFM(src, order=1, mode=0).vivtc.VDecimate(dryrun=True)
    if show_thr:
        return core.text.FrameProps(find, "VDecimateMaxBlockDiff")

    def _cb(n, f):
        if get_prop(f, "VDecimateMaxBlockDiff", int) > thr:
            frames.append(n)

    clip_async_render(find, progress="Searching 30 fps ranges...", callback=_cb)
    frames = rng(frames, min_length)
    frames = bookmarks(frames, name)
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

    frames = []
    find = core.vivtc.VFM(src, order=1, mode=2)

    def _cb(n, f):
        if get_prop(f, "_Combed", int) == 0:
            frames.append(n)

    clip_async_render(find, progress="Searching 60 fps ranges...", callback=_cb)
    frames = rng(frames, min_length)
    frames = bookmarks(frames, name)
    return find
