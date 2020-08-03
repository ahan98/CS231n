import numpy as np
from numpy.lib.stride_tricks import as_strided

def im2col_broadcast(x, FH, FW, stride):
    """
    Returns the columnized local regions for each batch data point as a 3-D
    tensor.

    Inputs:
    - x: (N,C,H,W) tensor (N batch size, C channels, H data height, W data width)
      NOTE: Assumes x has already been padded
    - FH: (int) filter height
    - FW: (int) filter width
    - stride: (int) horizontal step size between each locally filtered window
    - n_down: (int) number of windows that fit each image top-to-bottom
    - n_across: (int) number of windows that fit each image left-to-right

    Outputs:
    - im2col: (N, FH x FW x C, n_down, n_across), where n_down and n_across are
      the number of filters which fit across each image, accounting for padding
      and stride.
    """
    N, C, H, W = x.shape

    # max number of windows down and across (i.e. stride = 1)
    row_extent = H - FH + 1
    col_extent = W - FW + 1

    # Get batch block indices, i.e. the "flattened" indices for the left hand
    # corner of each image in the batch.
    # Example: X = (2,3,4,4) has batch block indices are 0 and 48.
    # Shape: (N,1,1)
    batch_idx = np.arange(N)[:, None, None] * C * H * W

    # Get indices of first block in first data point.
    # Example: if filter shape = 2x2, then the first block in X is made of
    # indices [0,1,4,5].
    # Shape: (1, FH, FW)
    start_idx = np.arange(FH)[None, :,None] * W + np.arange(FW)

    # Get first block indices for each channel
    # Shape: (C, FH, FW)
    didx = H * W * np.arange(C)
    start_idx = (didx[:,None] + start_idx.ravel()).reshape(-1, FH, FW)

    # Get indices for top-left corner of each local window in the first batch.
    # Shape: (1, row_extent, col_extent)
    # n_down and n_across are the number of local windows that actually fit
    # across each image, taking stride and padding into account.
    # Note that we don't literally compute them here (unlike in naive conv
    # implementation), but refer to them conceptually.
    offset_idx = np.arange(row_extent)[None, :, None] * W + np.arange(col_extent)

    # Reshape first block of first image into a (FH x FW) x 1 column
    first_block2col = start_idx.ravel()[None, :, None]
    # Broadcast the first block column across each image, so now we have the
    # columnized starting block for each image.
    first_batch2col = batch_idx + first_block2col
    # Apply stride to get the starting indices of each block in the first batch.
    # Unravel it (into a row vector), and broadcast it across the columnized
    # blocks for each image.
    # Shape: (N, FH x FW x C, n_down x n_across)
    im2col_indices = first_batch2col + offset_idx[:,::stride,::stride].ravel()

    return np.take(x, im2col_indices)


def im2col_stride(x, FH, FW, stride, n_down, n_across):
    N, C, H, W = x.shape

    # To calculate the strides, think of the number of array indices that must be
    # traversed to move one unit, where the unit is axis-dependent. Distance is
    # conceptualized as such, because ndarrays are stored contiguously in
    # memory.
    #
    # - Image <=> H*W*C, "distance" to the next image, within the same batch.
    # - Channel <=> H*W, distance to the next channel, within the same image.
    # - FH <=> W, distance to the next row, within the same filter.
    # - FW <=> 1, distance to the next column, within the same filter.
    # - n_down <=> stride * W, vertical distance to the next window, within the
    # same image.
    # - n_across <=> stride * 1, horizontal distance to the next window, within
    # the same image.
    shape = N, C, FH, FW, n_down, n_across
    strides = (H*W*C, H*W, W, 1, stride*W, stride)
    strides = x.itemsize * np.array(strides)

    x_col = as_strided(x, shape=shape, strides=strides)
    x_col = np.ascontiguousarray(x_col)
    x_col.shape = N, (C * FH * FW), (n_down * n_across)

    return x_col


def conv_forward_im2col(x, w, b, conv_param, method="stride"):
    """
    Returns the np.einsum() of the (N, FH x FW x C, n_down, n_across) im2col
    tensor and the similarly transformed weights (except each filter, which is
    composed of a weight matrix for each channel, is stretched into a single row,
    rather than column).

    Input:
    - x: Batch as (N,C,H,W) tensor
    - w: Filter weights of shape (F, C, FH, FW)
    - b: Biases of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Output:
    - out: (N, F, n_down, n_across) tensor
    - cache: (x, w, b, conv_param) tuple saving inputs for backprop
    """

    N, C, H, W = x.shape
    F, _, FH, FW = w.shape
    p = conv_param["pad"]
    stride = conv_param["stride"]

    # adjust image dimensions after padding
    H += 2 * p
    W += 2 * p
    assert (H - FH) % stride == 0
    assert (W - FW) % stride == 0
    n_down = int(1 + (H - FH) / stride)
    n_across = int(1 + (W - FW) / stride)
    pad_x = np.pad(x, ((0,0), (0,0), (p,p), (p,p)), "constant")

    if method == "stride":
        x_col = im2col_stride(pad_x, FH, FW, stride, n_down, n_across)
    elif method == "broadcast":
        x_col = im2col_broadcast(pad_x, FH, FW, stride)
    w_row = w.reshape(F,-1)

    out = np.einsum("fb,nbw->nfw", w_row, x_col)
    out.shape = (N, F, n_down, n_across)
    out += b[:, None, None]
    cache = (x, w, b, conv_param)

    return out, cache
