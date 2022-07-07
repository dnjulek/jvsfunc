"""
Helper functions
"""

from __future__ import annotations

from typing import List, Sequence
from math import floor, ceil
import vapoursynth as vs
core = vs.core


def plane_stats(src: vs.VideoNode, lower_thr: float = 0, upper_thr: float = 0, cuda: bool = False) -> vs.VideoNode:
    """
    Settable PlaneStats func.

    :param src: Input clip.
    :param lower_thr: Define the ratio of pixel number to determine the Floor of value range.
    :param upper_thr: Define the ratio of pixel number to determine the Ceil of value range.
    :cuda: Enables cupy.
    """

    if (lower_thr + upper_thr) >= 1:
        raise ValueError('plane_stats: (lower_thr + upper_thr) cannot be >= 1.')

    if (lower_thr == 0) and (upper_thr == 0):
        return src.std.PlaneStats()

    is_float = src.format.sample_type == vs.FLOAT
    total = src.height * src.width
    minthr = floor((total - 1) * lower_thr)
    maxthr = ceil((total - 1) * (1 - upper_thr))

    def process(n, f: vs.VideoFrame):
        if cuda:
            import cupy as cp
            arr = cp.array(f[0])
            arr = cp.reshape(arr, (total))
            arr = cp.sort(arr)
        else:
            import numpy as np
            arr = np.array(f[0])
            arr = np.reshape(arr, (total))
            arr = np.sort(arr)

        pmin = float(arr[minthr]) if is_float else int(arr[minthr])
        pmax = float(arr[maxthr]) if is_float else int(arr[maxthr])

        fout = f.copy()
        fout.props['PlaneStatsMin'] = pmin
        fout.props['PlaneStatsMax'] = pmax
        return fout

    return src.std.ModifyFrame(src, process)


def _bookmarks(flist: List[int], name: str):
    name += '.bookmarks'
    frames = str(flist)[1:-1]
    text_file = open(name, 'w')
    text_file.write(frames)
    text_file.close()


def _rng(flist1: List[int], min_length: int):
    flist2: List = []
    flist3: List = []
    prev_n = -1

    for n in flist1:
        if prev_n+1 != n:
            if flist3:
                flist2.append(flist3)
                flist3 = []
        flist3.append(n)
        prev_n = n

    if flist3:
        flist2.append(flist3)

    flist4 = [i for i in flist2 if len(i) > min_length]
    first_frame = [i[0] for i in flist4]
    last_frame = [i[-2] for i in flist4]
    final = first_frame + last_frame
    final[::2] = first_frame
    final[1::2] = last_frame
    return final


def _ex_matrix(r: int = 1):
    b = [i for i in range(-1*r, r+1)]
    matrix = [f'x[{x},{y}] ' for x in b for y in b]
    matrix.pop(len(matrix)//2)
    return ''.join(matrix)


def _morpho_matrix(size: int = 2, mm: str = 'max'):
    is_even = size % 2 == 0
    rd = size // 2
    mt = [i for i in range(-1*rd, rd+1)]

    odd = [f'x[{x},{y}] {mm} ' for x in mt for y in mt]
    even1 = [f'x[{x},{y}] {mm} ' for x in mt[:-1] for y in mt[:-1]]
    even2 = [f'x[{x},{y}] {mm} ' for x in mt[-1:] for y in mt[:-2]]
    matrix = even1 + even2 if is_even else odd
    matrix = ''.join(matrix)  # type:ignore
    return matrix[:8] + matrix[12:]


def _ex_planes(src: vs.VideoNode, expr: List[str], planes: int | Sequence[int] | None = None) -> List[str]:

    if planes is not None:
        plane_range = range(src.format.num_planes)
        planes = [planes] if isinstance(planes, int) else planes
        expr = [expr[0] if i in planes else '' for i in plane_range]

    return expr
