"""
Helper functions
"""

from typing import List
import vapoursynth as vs
core = vs.core


def bookmarks(flist: List[int], name: str):
    name += '.bookmarks'
    frames = str(flist)[1:-1]
    text_file = open(name, 'w')
    text_file.write(frames)
    text_file.close()


def rng(flist1: List[int], min_length: int):
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


def ex_matrix(r: int = 1):
    b = [i for i in range(-1*r, r+1)]
    matrix = [f'x[{x},{y}] ' for x in b for y in b]
    matrix.pop(len(matrix)//2)
    return ''.join(matrix)


def morpho_matrix(size: int = 2, mm: str = 'max'):
    is_even = size % 2 == 0
    rd = size // 2
    mt = [i for i in range(-1*rd, rd+1)]

    odd = [f'x[{x},{y}] {mm} ' for x in mt for y in mt]
    even1 = [f'x[{x},{y}] {mm} ' for x in mt[:-1] for y in mt[:-1]]
    even2 = [f'x[{x},{y}] {mm} ' for x in mt[-1:] for y in mt[:-2]]
    matrix = even1 + even2 if is_even else odd
    matrix = ''.join(matrix)  # type:ignore
    return matrix[:8] + matrix[12:]


def inter_pattern(clipa: List[vs.VideoNode], clipb: List[vs.VideoNode]):
    inter0 = core.std.Interleave([clipb[0], clipa[1], clipa[2], clipa[3], clipa[4]])
    inter1 = core.std.Interleave([clipa[0], clipb[1], clipa[2], clipa[3], clipa[4]])
    inter2 = core.std.Interleave([clipa[0], clipa[1], clipb[2], clipa[3], clipa[4]])
    inter3 = core.std.Interleave([clipa[0], clipa[1], clipa[2], clipb[3], clipa[4]])
    inter4 = core.std.Interleave([clipa[0], clipa[1], clipa[2], clipa[3], clipb[4]])
    return [inter0, inter1, inter2, inter3, inter4]


def jdeblend_eval(n: int, f: List[vs.VideoFrame], src: vs.VideoNode, inters: List[vs.VideoNode]):
    comb = [f[i].props['_Combed'] for i in [0, 1]]
    pattern = n % 5
    if comb[0] == 1:
        src = inters[pattern]
    return src[n+1] if sum(comb) == 2 else src
