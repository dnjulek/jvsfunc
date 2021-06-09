from vsutil import get_y, depth, iterate
from math import sqrt
import vapoursynth as vs
core = vs.core

def dehalo_mask(src, expand=0.5, iterations=2):
    """
    Based on muvsfunc.YAHRmask(), stand-alone version with some tweaks.

    Parameters:
    src:        Input clip. I suggest to descale (if possible) and nnedi3_rpow2 first, for a cleaner mask.
    expand:     Expansion of edge mask. Default is 0.5.
    iterations: Protects parallel lines and corners that are usually damaged by YAHR. Default is 2.
    """
    is8 = src.format.bits_per_sample == 8
    src_b = src if is8 else depth(src, 8)#8 bit only because the result in 16 is the same but slower
    luma = get_y(src_b)
    vEdge = core.std.Expr([luma, luma.std.Maximum().std.Maximum()], ['y x - 8 - 128 *'])
    mask1 = vEdge.tcanny.TCanny(sigma=sqrt(expand*2), mode=-1).std.Expr(['x 16 *'])
    mask2 = iterate(vEdge, core.std.Maximum, iterations)
    mask2 = iterate(mask2, core.std.Minimum, iterations)
    mask2 = mask2.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Invert()
    mask = core.std.Expr([mask1, mask2], ['x y min'])
    return mask if is8 else depth(mask, src.format.bits_per_sample)

def FreezeFramesMod(src, mode=1, ranges=None, single=False):
    """
    Mod to easily replace frames with the previous or next one.

    Parameters:
    src:    Input clip.
    mode:   Frame to be copied, 1 or 'prev' for previous one, 2 or 'next' for next one. Default is 1.
    ranges: List of frame ranges to be processed.
    single: If True, each frame in the list will be processed individually (like a keyframe list), if False, it will be taken as ranges. Default is False.
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

def find_comb(src, name='comb_list'):
    """
    Creates a VSEdit Bookmarks file with a list of combed frames, to be used in scene filtering.

    Parameters:
    src:    Input clip.
    name:   Output file name. Default is 'comb_list'.
    """
    name += '.bookmarks'
    src = depth(src, 8) if src.format.bits_per_sample != 8 else src
    find = core.tdm.IsCombed(src)
    diff = core.std.PlaneStats(find)
    frames = [i for i,f in enumerate(diff.frames()) if f.props._Combed == 1]
    frames = str(frames)[1:-1]
    text_file = open(name, 'w')
    text_file.write(frames)
    text_file.close()

def find_60p(src, min_length=60, name='60p_ranges'):
    """
    Creates a VSEdit Bookmarks file with possible 60 fps ranges from a VFR (interlaced) clip, to be used in scene filtering.

    Parameters:
    src:        Input clip. Has to be 60i (30000/1001).
    min_length: Non 60 fps consecutive frames needed to end a range.
    name:       Output file name. Default is '60p_ranges'.
    """
    name +='.bookmarks'
    if src.fps_num != 30000 or src.fps_den != 1001:
        raise ValueError('find_60p: This script can only be used with 60i clips.')
    
    src = core.vivtc.VFM(src, order=1, mode=2)
    frames = [i for i,f in enumerate(src.frames()) if f.props._Combed == 0]
    a = frames
    b = []
    subList = []
    prev_n = -1

    for n in a:
        if prev_n+1 != n:
            if subList:
                b.append(subList)
                subList = []
        subList.append(n)
        prev_n = n

    if subList:
        b.append(subList)

    frames2 = [i for i in b if len(i) > min_length]
    first_frame = [i[0] for i in frames2]
    last_frame = [i[-2] for i in frames2]
    final = first_frame + last_frame
    final[::2] = first_frame
    final[1::2] = last_frame

    sfinal = str(final)[1:-1]
    text_file = open(name,'w')
    text_file.write(sfinal)
    text_file.close()