"""
Deinterlace functions
"""

from __future__ import annotations

import vapoursynth as vs
core = vs.core


def decimate(src: vs.VideoNode, pattern: int = 0) -> vs.VideoNode:
    """
    :param src: Input clip.
    :param pattern: Decimate pattern.
    """
    selectlist = [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]]
    return src.std.SelectEvery(5, selectlist[pattern % 5])
