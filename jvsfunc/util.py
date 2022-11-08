"""
Util functions
"""

from __future__ import annotations

from typing import List, Tuple
import vapoursynth as vs
core = vs.core

Range = int | None | Tuple[int | None, int | None]


def rfs(clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: Range | list[Range] | None,
        exclusive: bool = False) -> vs.VideoNode:
    """
    Replace frames.

    :param clip_a:          Original clip.
    :param clip_b:          Replacement clip.
    :param ranges:          Ranges to replace clip_a with clip_b.
    :param exclusive:       Use exclusive ranges.
    """
    if ranges != 0 and not ranges:
        return clip_a

    def normalize_ranges(clip: vs.VideoNode, ranges: Range | List[Range]) -> List[Tuple[int, int]]:
        ranges = ranges if isinstance(ranges, list) else [ranges]
        out = []
        for r in ranges:
            if isinstance(r, tuple):
                start, end = r
                if start is None:
                    start = 0
                if end is None:
                    end = clip.num_frames - 1
            elif r is None:
                start = clip.num_frames - 1
                end = clip.num_frames - 1
            else:
                start = r
                end = r
            if start < 0:
                start = clip.num_frames - 1 + start
            if end < 0:
                end = clip.num_frames - 1 + end
            out.append((start, end))
        return out

    def to_list(list_in: List[Tuple[int, int]]) -> List[int]:
        shift = 1 + exclusive
        out = []
        for x in list_in:
            for y in range(x[0], x[1] + shift):
                out.append(y)
        return out

    nranges = normalize_ranges(clip_b, ranges)
    flist = to_list(nranges)

    fmax = max(flist)
    blen = clip_b.num_frames
    if (fmax >= blen):
        raise ValueError(f"rfs: clip_b can't replace frame {fmax}, it has only {blen} frames.")

    return core.replaceframes.RFS(clip_a, clip_b, flist)
