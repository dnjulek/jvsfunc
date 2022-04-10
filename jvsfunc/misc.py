"""
Miscellaneous functions
"""

from functools import partial
from typing import List, Optional
import vapoursynth as vs
core = vs.core


def FreezeFramesMod(src: vs.VideoNode, mode: str = 'prev', ranges: List[int] = [],
                    single: bool = False) -> vs.VideoNode:
    """
    Mod to easily replace frames with the previous or next one.

    :param src: Input clip.
    :param mode: Frame to be copied, 1 or 'prev' for previous one, 2 or 'next' for next one.
    :param ranges: List of frame ranges to be processed.
    :param single: If True, each frame in the list will be processed individually (like a keyframe list),
                   if False, it will be taken as ranges.
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


def replace_keyframe(src: vs.VideoNode, thr: float = 0.30, kf_list: Optional[List[int]] = None,
                     show_thr: bool = False) -> vs.VideoNode:
    """
    Replace the frame after a scene change with the next frame. Helps to fix broken keyframes.

    :param src: Input clip.
    :param thr: Threshold is the difference between the two frames, it must be high enough to ignore artifacts
                and low enough to avoid freezing moving scenes.
    :param kf_list: List of keyframes, can be made with lvsfunc.render.find_scene_changes(clip, scxvid=True),
                    if it is not provided, scxvid will generate the keyframes in real time,
                    which can be bad for the preview.
    :param show_thr: Shows the threshold of the frame. This also disable the replacement
                     for a better frame seeking in preview.
    """
    def _show_thr(n: int, f: vs.VideoFrame, clip: vs.VideoNode):
        diff = str(f.props['PlaneStatsDiff'] * 100)  # type:ignore
        return clip.sub.Subtitle("Frame " + str(n) + " of " + str(clip.num_frames) + f"\nCurrent thr: {diff}")

    def _schang_xvid(n: int, f: vs.VideoFrame, clip: vs.VideoNode, thr: float):
        if f.props['_SceneChangePrev'] == 1 and f.props['PlaneStatsDiff'] * 100 < thr:  # type:ignore
            return clip[n+1]
        else:
            return clip

    def _schang_list(n: int, f: vs.VideoFrame, clip: vs.VideoNode, thr: float, klist: List[int]):
        if n in klist and f.props['PlaneStatsDiff'] * 100 < thr:  # type:ignore
            return clip[n+1]
        else:
            return clip

    diff = core.std.PlaneStats(src, src[1:])

    if show_thr:
        return core.std.FrameEval(src, partial(_show_thr, clip=src), prop_src=diff)

    if kf_list is None:
        xvid = diff.resize.Bilinear(640, 360, format=vs.YUV420P8).scxvid.Scxvid()
        return core.std.FrameEval(src, partial(_schang_xvid, clip=src, thr=thr), prop_src=xvid)

    return core.std.FrameEval(src, partial(_schang_list, clip=src, thr=thr, klist=kf_list), prop_src=diff)