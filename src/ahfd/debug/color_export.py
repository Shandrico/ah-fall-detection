"""Export the colour stream of a depth .bag/.db3 to an .mp4 for labelling.

Lives in ``ahfd.debug`` because it writes raw imagery (cv2.VideoWriter), which
the privacy guard permits only in this package. A depth .bag/.db3 already holds
the consented staged frames; this re-encodes just the colour so the posture
labeller -- which reads an .mp4, not a .db3 -- can scrub it. Delete the .mp4
once the clip is labelled, the same as the .bag.
"""

from __future__ import annotations

from pathlib import Path


def export_color(bag_path, out_path, *, preview: bool = False) -> int:
    """Read colour frames from a depth .bag/.db3 and write them to an .mp4.

    Returns the frame count. Colour only (``with_depth=False``), so there is no
    depth align/filter cost and it runs fast. The output fps is the recording's
    nominal rate; the labeller's segment times still line up with the extracted
    tracks because ``build_dataset`` normalises each clip's track timestamps to
    start at zero.
    """
    import cv2

    from ahfd.capture.realsense import BagSource

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = BagSource(str(bag_path), with_depth=False)  # colour only -> fast
    fps = float(src.meta.fps or 30.0)
    writer = None
    n = 0
    try:
        for frame in src:
            img = frame.bgr
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(img)
            n += 1
            if preview:
                disp = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
                cv2.imshow("ahfd export-color", disp)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        src.close()
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
    return n
