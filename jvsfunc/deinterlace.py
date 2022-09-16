"""
Deinterlace functions
"""

from __future__ import annotations
from typing import List

from functools import partial
from .mask import comb_mask
import vapoursynth as vs
core = vs.core


def decimate(src: vs.VideoNode, pattern: int = 0) -> vs.VideoNode:
    """
    :param src: Input clip.
    :param pattern: Decimate pattern.
    """
    selectlist = [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]]
    return src.std.SelectEvery(5, selectlist[pattern % 5])


def select_lesscombed(src: List[vs.VideoNode], cthresh=6, mthresh=0, metric=0) -> vs.VideoNode:
    """
    :param src: List of clips to select the less combed.
    :param cthresh: Spatial combing threshold.
    :param mthresh: Motion adaptive threshold.
    :param metric: Sets which spatial combing metric is used to detect combed pixels.
    """
    mask = [comb_mask(i, cthresh=cthresh, mthresh=mthresh, metric=metric, planes=[0]).std.PlaneStats() for i in src]

    def _select(n: int, f: List[vs.VideoFrame], srcs: List[vs.VideoNode]) -> vs.VideoNode:
        avg = [f[i].props["PlaneStatsAverage"] for i in [0, 1, 2, 3]]
        return srcs[avg.index(min(avg))]  # type:ignore

    return core.std.FrameEval(src[0], partial(_select, srcs=src), mask)
