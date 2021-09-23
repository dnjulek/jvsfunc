from lvsfunc.render import clip_async_render
from lvsfunc.util import get_prop
from vsutil import get_y, depth, iterate
from functools import partial
from math import sqrt
import vapoursynth as vs
core = vs.core

def expr_dilate(src, size = 2):
    """
    Same result as core.morpho.Dilate(), faster and workable in 32 bit.
    """
    expr = _expr_matrix(size, mm = 'max')
    return core.akarin.Expr(src, expr)

def expr_erode(src, size = 2):
    """
    Same result as core.morpho.Erode(), faster and workable in 32 bit.
    """
    expr = _expr_matrix(size, mm = 'min')
    return core.akarin.Expr(src, expr)

def expr_close(src, size = 2):
    """
    Same result as core.morpho.Close(), faster and workable in 32 bit.
    """
    close = expr_dilate(src, size)
    return expr_erode(close, size)

def expr_open(src, size = 2):
    """
    Same result as core.morpho.Open(), faster and workable in 32 bit.
    """
    close = expr_erode(src, size)
    return expr_dilate(close, size)

def dehalo_mask(src, expand=0.5, iterations=2, brz=255):
    """
    Based on muvsfunc.YAHRmask(), stand-alone version with some tweaks.

    Parameters:
    clip    src:            Input clip. I suggest to descale (if possible) and nnedi3_rpow2 first, for a cleaner mask.
    float   expand (0.5):   Expansion of edge mask.
    int     iterations (2): Protects parallel lines and corners that are usually damaged by YAHR.
    int     brz (255):      Adjusts the internal line thickness.
    """
    if brz > 255 or brz < 0:
        raise ValueError('dehalo_mask: brz must be between 0 and 255.')
    
    is8 = src.format.bits_per_sample == 8
    src_b = src if is8 else depth(src, 8)#8 bit only because the result in 16 is the same but slower
    luma = get_y(src_b)
    vEdge = core.std.Expr([luma, luma.std.Maximum().std.Maximum()], ['y x - 8 - 128 *'])
    mask1 = vEdge.tcanny.TCanny(sigma=sqrt(expand*2), mode=-1).std.Expr(['x 16 *'])
    mask2 = iterate(vEdge, core.std.Maximum, iterations)
    mask2 = iterate(mask2, core.std.Minimum, iterations)
    mask2 = mask2.std.Invert().std.Binarize(80)
    mask3 = mask2.std.Inflate().std.Inflate().std.Binarize(brz)
    mask4 = mask3 if brz < 255 else mask2
    mask4 = mask4.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    mask = core.std.Expr([mask1, mask4], ['x y min'])
    return mask if is8 else depth(mask, src.format.bits_per_sample)

def FreezeFramesMod(src, mode='prev', ranges=None, single=False):
    """
    Mod to easily replace frames with the previous or next one.

    Parameters:
    clip    src:            Input clip.
    str     mode ('prev'):  Frame to be copied, 1 or 'prev' for previous one, 2 or 'next' for next one.
    list    ranges:         List of frame ranges to be processed.
    bool    single (False): If True, each frame in the list will be processed individually (like a keyframe list), if False, it will be taken as ranges.
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

def replace_keyframe(src, thr=0.30, keyframe_list=None, show_thr=False):
    """
    Replace the frame after a scene change with the next frame. Helps to fix broken keyframes.

    Parameters:
    clip    src:                        Input clip.
    float   thr (0.30):                 Threshold is the difference between the two frames, it must be high enough to ignore artifacts 
                                        and low enough to avoid freezing moving scenes.
    list    keyframe_list (Optional):   List of keyframes, can be made with lvsfunc.render.find_scene_changes(clip, scxvid=True),
                                        if it is not provided, scxvid will generate the keyframes in real time, which can be bad for the preview.
    bool    show_thr (False):           Shows the threshold of the frame. This also disable the replacement for a better frame seeking in preview.
    """
    def _show_thr(n, f, clip):
        return clip.sub.Subtitle("Frame " + str(n) + " of " + str(clip.num_frames) + "\nCurrent thr: {}".format(str(f.props.PlaneStatsDiff * 100)))

    def _schang_xvid(n, f, clip, thr):
        if f.props._SceneChangePrev == 1 and f.props.PlaneStatsDiff * 100 < thr:
            return clip[n+1]
        else:
            return clip
    
    def _schang_list(n, f, clip, thr, klist):
        if n in klist and f.props.PlaneStatsDiff * 100 < thr:
            return clip[n+1]
        else:
            return clip

    diff = core.std.PlaneStats(src, src[1:])

    if show_thr:
        return core.std.FrameEval(src, partial(_show_thr, clip=src), prop_src=diff)
    
    if keyframe_list is None:
        xvid = diff.resize.Bilinear(640, 360, format=vs.YUV420P8).scxvid.Scxvid()
        return core.std.FrameEval(src, partial(_schang_xvid, clip=src, thr=thr), prop_src=xvid)
    
    return core.std.FrameEval(src, partial(_schang_list, clip=src, thr=thr, klist=keyframe_list), prop_src=diff)

def JIVTC_Deblend(src, pattern, chroma_only=True, tff=True):
    """
    fvsfunc.JIVTC() modified to use a deblend based on lvsfunc instead of original bobber (yadifmod).

    JIVTC_Deblend works similar to the original, and follows the same A, AB, BC, C, D pattern.
    This function should only be used when a normal ivtc or ivtc+bobber leaves chroma blend to a every fourth frame.
    You can disable chroma_only to use in luma as well, but it is not recommended.
    
    Parameters:
    clip   src:                  Source clip. Has to be 60i (30000/1001).
    int    pattern:              First frame of any clean-combed-combed-clean-clean sequence.
    bool   chroma_only (True):   If set to False, luma will also receive deblend process.
    bool   tff (True):           Set top field first (True) or bottom field first (False).
    """
    if src.fps_num != 30000 or src.fps_den != 1001:
        raise ValueError('JIVTC_Deblend: This filter can only be used with 60i clips.')
    
    pattern = pattern % 5
    def deblend(src):
        blends_a = range(1, src.num_frames - 1, 5)
        blends_b = range(2, src.num_frames - 1, 5)

        def calculate(n, f, src):
            if n % 5 in [0, 3, 4]:
                return src
            else:
                if n in blends_a:
                    a, ab, bc, c = src[n - 1], src[n], src[n + 1], src[n + 2]
                    dbd = core.std.Expr([a, ab, bc, c], "z a 2 / - y x 2 / - +")
                    dbd = dbd.vinverse.Vinverse()
                    return dbd if f.props._Combed == 1 else src
                return src
        
        dbd = core.std.FrameEval(src, partial(calculate, src=src), src.tdm.IsCombed())
        return core.std.DeleteFrames(dbd, blends_b).std.AssumeFPS(fpsnum=24000, fpsden=1001)
	
    defivtc = core.std.SeparateFields(src, tff=tff).std.DoubleWeave()
    selectlist = [[0,3,6,8], [0,2,5,8], [0,2,4,7], [2,4,6,9], [1,4,6,8]]
    ivtced = core.std.SelectEvery(defivtc, 10, selectlist[pattern])
	
    selectlist = [deblend(src), ivtced[:1]+deblend(src[1:]), ivtced[:2]+deblend(src[2:]), ivtced[:2]+deblend(src[3:]), ivtced[:3]+deblend(src[4:])]
    deblended = selectlist[pattern]
	
    inter = core.std.Interleave([ivtced, deblended])
    selectlist = [[0,3,4,6], [0,2,5,6], [0,2,4,7], [0,2,4,7], [1,2,4,6]]
    final = core.std.SelectEvery(inter, 8, selectlist[pattern])
	
    final_y = core.std.ShufflePlanes([ivtced, final, final], [0, 1, 2], vs.YUV)
    final = final_y if chroma_only else final
    return core.std.SetFrameProp(final, prop='_FieldBased', intval=0)

def jdeblend(src_fm, src, vinverse=True):
    """
    Automatically deblends if normal field matching leaves 2 blends every 5 frames (like avs's ExBlend).

    Parameters:
    clip   src_fm:              Source after field matching, must have field=3 and low cthresh.
    clip   src:                 Untouched source.
    bool   vinverse (True):     Vinverse for post processing.

    Example:
    src = 
    vfm = src.vivtc.VFM(order=1, field=3, cthresh=3)
    dblend = jdeblend(vfm, src)
    dblend = core.std.ShufflePlanes([vfm, dblend], [0, 1, 2], vs.YUV)
    dblend = jdeblend_kf(dblend, vfm)
    """

    a, ab, bc, c = src[0] + src[:-1], src, src[1:] + src[-1], src[2:] + src[-2:]
    dbd = core.std.Expr([a, ab, bc, c], "z a 2 / - y x 2 / - +")
    select_src = [src.std.SelectEvery(5, i) for i in range(5)]
    select_dbd = [dbd.std.SelectEvery(5, i) for i in range(5)]
    inter0 = core.std.Interleave([select_dbd[0], select_src[1], select_src[2], select_src[3], select_src[4]])
    inter1 = core.std.Interleave([select_src[0], select_dbd[1], select_src[2], select_src[3], select_src[4]])
    inter2 = core.std.Interleave([select_src[0], select_src[1], select_dbd[2], select_src[3], select_src[4]])
    inter3 = core.std.Interleave([select_src[0], select_src[1], select_src[2], select_dbd[3], select_src[4]])
    inter4 = core.std.Interleave([select_src[0], select_src[1], select_src[2], select_src[3], select_dbd[4]])

    def calculate(n, f, src, inters):
        comb = [f[i].props['_Combed'] for i in [0, 1]]
        pattern = n % 5
        if comb[0] == 1:
            src = inters[pattern]
        return src[n+1] if sum(comb) == 2 else src
       
    inters = [inter0, inter1, inter2, inter3, inter4]
    inters = [core.vinverse.Vinverse(i) for i in inters] if vinverse else inters
    return core.std.FrameEval(src_fm, partial(calculate, src=src_fm, inters=inters), [src_fm, src_fm[0]+src_fm[:-1]])

def jdeblend_bob(src_fm, bobbed):
    """
    Stronger version of jdeblend() that uses a bobbed clip to deblend.

    Parameters:
    clip   src_fm:              Source after field matching, must have field=3 and low cthresh.
    clip   src:                 Bobbed source.

    Example:
    src = 
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
    inter0 = core.std.Interleave([select_dbd[0], select_src[1], select_src[2], select_src[3], select_src[4]])
    inter1 = core.std.Interleave([select_src[0], select_dbd[1], select_src[2], select_src[3], select_src[4]])
    inter2 = core.std.Interleave([select_src[0], select_src[1], select_dbd[2], select_src[3], select_src[4]])
    inter3 = core.std.Interleave([select_src[0], select_src[1], select_src[2], select_dbd[3], select_src[4]])
    inter4 = core.std.Interleave([select_src[0], select_src[1], select_src[2], select_src[3], select_dbd[4]])

    def calculate(n, f, src, inters):
        comb = [f[i].props['_Combed'] for i in [0, 1]]
        pattern = n % 5
        if comb[0] == 1:
            src = inters[pattern]
        return src[n+1] if sum(comb) == 2 else src
       
    inters = [inter0, inter1, inter2, inter3, inter4]
    return core.std.FrameEval(src_fm, partial(calculate, src=src_fm, inters=inters), [src_fm, src_fm[0]+src_fm[:-1]])

def jdeblend_kf(src, src_fm):
    """
    Should be used after jdeblend() to fix scene changes.

    Parameters:
    clip   src:                 Source after jdeblend().
    clip   src_fm:              Source after field matching, must have field=3 and low cthresh.
    """

    def keyframe(n, f, src):
        keyfm = [f[i].props['VFMSceneChange'] for i in [0, 1]]
        kf_end = keyfm[0] > keyfm[1]
        kf_start = sum(keyfm) == 2
        is_cmb = f[0].props['_Combed'] == 1
        src = src[n-1] if kf_end and is_cmb else src
        return src[n+1] if kf_start and is_cmb else src

    return core.std.FrameEval(src, partial(keyframe, src=src), [src_fm, src_fm[0]+src_fm[:-1]])

def find_comb(src, name='comb_list'):
    """
    Creates a VSEdit Bookmarks file with a list of combed frames, to be used in scene filtering.

    Parameters:
    clip    src:                Input clip.
    str     name ('comb_list'): Output file name.
    """
    frames = []
    src = depth(src, 8) if src.format.bits_per_sample != 8 else src
    find = core.tdm.IsCombed(src).std.PlaneStats()

    def _cb(n, f):
        if get_prop(f, "_Combed", int) == 1:
            frames.append(n)

    clip_async_render(find, progress="Searching combed...", callback=_cb)
    frames = _bookmarks(frames, name)
    return find

def find_30p(src, min_length=34, thr=2000, name='30p_ranges', show_thr=False):
    """
    Creates a VSEdit Bookmarks file with possible 30 fps ranges from a VFR (interlaced) clip, to be used in scene filtering.

    Parameters:
    clip    src:                    Input clip. Has to be 60i (30000/1001).
    int     min_length (34):        Non 30 fps consecutive frames needed to end a range.
    int     thr (2000):             Threshold for the frame not to be considered a duplicate.
    str     name ('30p_ranges'):    Output file name.
    bool    show_thr (False):       Shows the threshold of the frame.
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
    frames = _rng(frames, min_length)
    frames = _bookmarks(frames, name)
    return find

def find_60p(src, min_length=60, name='60p_ranges'):
    """
    Creates a VSEdit Bookmarks file with possible 60 fps ranges from a VFR (interlaced) clip, to be used in scene filtering.

    Parameters:
    clip    src:                    Input clip. Has to be 60i (30000/1001).
    int     min_length (60):        Non 60 fps consecutive frames needed to end a range.
    str     name ('60p_ranges'):    Output file name.
    """
    if src.fps_num != 30000 or src.fps_den != 1001:
        raise ValueError('find_60p: This script can only be used with 60i clips.')
    
    frames = []
    find = core.vivtc.VFM(src, order=1, mode=2)

    def _cb(n, f):
        if get_prop(f, "_Combed", int) == 0:
            frames.append(n)

    clip_async_render(find, progress="Searching 60 fps ranges...", callback=_cb)
    frames = _rng(frames, min_length)
    frames = _bookmarks(frames, name)
    return find

def _bookmarks(flist, name):
    name += '.bookmarks'
    frames = str(flist)[1:-1]
    text_file = open(name, 'w')
    text_file.write(frames)
    text_file.close()

def _rng(flist1, min_length):
    flist2 = []
    flist3 = []
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

def _expr_matrix(size = 2, mm = 'max'):

    is_even = size % 2 == 0
    rd = size // 2
    r1 = [i for i in range(rd + 1)]
    r2 = [i * -1 for i in r1[1:]]
    r2.reverse()
    r3 = r2 + r1

    odd = [f'x[{x},{y}] {mm} ' for x in r3 for y in r3]
    even1 = [f'x[{x},{y}] {mm} ' for x in r3[:-1] for y in r3[:-1]]
    even2 = [f'x[{x},{y}] {mm} ' for x in r3[-1:] for y in r3[:-2]]
    matrix = even1 + even2 if is_even else odd
    matrix = ''.join(matrix)
    return matrix[:8] + matrix[12:]