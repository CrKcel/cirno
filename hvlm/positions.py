"""M-RoPE helpers that need only numpy (so they can be unit-tested without torch).

Layout follows Qwen3-VL / Qwen3.5 *interleaved* M-RoPE: rotary frequency index f is
assigned to the temporal / height / width axis in a T,H,W,T,H,W,... pattern, and
the remaining high-index frequencies all belong to the temporal axis.
"""
import numpy as np


def mrope_axis_map(mrope_section):
    """Return an int array of length sum(section): the position axis (0=t,1=h,2=w) used by each frequency."""
    n = int(sum(mrope_section))
    axis = np.zeros(n, dtype=np.int64)
    for a in (1, 2):
        length = mrope_section[a] * 3
        axis[a:length:3] = a
    return axis


def build_mrope_positions(input_ids, image_token_id, grid):
    """Position ids for one sequence with fixed-size images.

    input_ids: 1-D int array (length T).
    grid: merged patches per image side (e.g. 7 for 224px / 16px patches / 2x2 merge).
          Each image is a run of exactly grid*grid `image_token_id` tokens.
    Returns int64 array [3, T] with rows (temporal, height, width).

    Text tokens use the same running position on all three axes.  The tokens of an
    image share one temporal position and take (row, col) offsets for h / w; the text
    that follows resumes at start + max(grid_h, grid_w), as in Qwen2-VL / Qwen3-VL.
    """
    ids = np.asarray(input_ids)
    T = ids.shape[0]
    pos = np.zeros((3, T), dtype=np.int64)
    n_img = grid * grid
    p = 0
    i = 0
    while i < T:
        if ids[i] == image_token_id:
            j = min(i + n_img, T)
            run = np.arange(j - i)
            pos[0, i:j] = p
            pos[1, i:j] = p + run // grid
            pos[2, i:j] = p + run % grid
            p += grid  # max(grid_h, grid_w)
            i = j
        else:
            pos[:, i] = p
            p += 1
            i += 1
    return pos
