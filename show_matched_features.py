import numpy as np
import cv2 as cv

def padarray(array, pad_width, pad_value=0, pad_position='pre') -> None:
    """Pads a given array with zeros

    Args:
        array (_type_): numpy array to be padded.
        pad_width (_type_): padding width, a tuple indicating the number of rows and columns to pad.
        pad_value (int, optional): pad value, the value to use for padding. Defaults to 0.
        pad_position (str, optional): Whether to pad 'pre' (before) or 'post' (after) the array.. Defaults to 'pre'.

    Raises:
        ValueError: _description_

    Returns:
        None: _description_
    """
    if pad_position == 'pre':
        return np.pad(array, ((pad_width[0], 0), (pad_width[1], 0)), mode='constant', constant_values=pad_value)
    elif pad_position == 'post':
        return np.pad(array, ((0, pad_width[0]), (0, pad_width[1])), mode='constant', constant_values=pad_value)
    else:
        raise ValueError("pad_position must be 'pre' or 'post'")
    
def show_matched_features(
    I1: np.array,
    kp1: np.float32,
    I2: np.array,
    kp2: np.float32,
    matches: np.array,
    circ_radius: float=2,
    line_thickness: float=1,
) -> np.ndarray:
    """_summary_

    Args:
        I1 (np.array): input image 1
        kp1 (np.float32): keypoints matching image 1
        I2 (np.array): input image 2
        kp2 (np.float32): keypoints matching image 2
        matches (np.array): matches vector
        circ_radius (float, optional): keypoint radius. Defaults to 1.
        line_thickness (float, optional): line width. Defaults to 1.

    Returns:
        np.ndarray: output image
    """
    # Pad the smaller image
    paddedSize = [max(I1.shape[0], I2.shape[0]), max(I1.shape[1], I2.shape[1])]
    I1pad = [paddedSize[0] - I1.shape[0], paddedSize[1] - I1.shape[1]]
    I2pad = [paddedSize[0] - I2.shape[0], paddedSize[1] - I2.shape[1]]
    I1pre = np.round(np.divide(I1pad, 2)).astype(int).tolist()
    I2pre = np.round(np.divide(I2pad, 2)).astype(int).tolist()
    I1 = padarray(I1, I1pre, 0, 'pre')
    I2 = padarray(I2, I2pre, 0, 'pre')
    I1 = padarray(I1, np.subtract(I1pad, I1pre), 0, 'post')
    I2 = padarray(I2, np.subtract(I2pad, I2pre), 0, 'post')
    
    # Fuse the images
    if len(I1.shape) == 3:
        imfused = np.dstack((I2[:, :, 0], I1[:, :, 0], I2[:, :, 0]))
    else:
        imfused = np.dstack((I2, I1, I2))

    # Offset the matched keypoints
    offset1 = np.flip(I1pre).tolist()
    offset2 = np.flip(I2pre).tolist()

    kp1 = np.vstack(kp1.tolist())
    kp2 = np.vstack(kp2.tolist())

    kp1 = np.add(kp1, offset1)
    kp2 = np.add(kp2, offset2)

    # Plot the matches
    for idx, m in enumerate(matches):
        # So the keypoint locs are stored as a tuple of floats.  cv.line(), like most other things,
        # wants locs as a tuple of ints.
        if m == 1:
            end1 = tuple(np.round(kp1[idx, :]).astype(int))
            end2 = tuple(np.round(kp2[idx, :]).astype(int))
            cv.line(imfused, end1, end2, (255, 255, 0), line_thickness, cv.LINE_AA)
            cv.circle(
                imfused, end1, circ_radius, (0, 255, 0), line_thickness, cv.LINE_AA
            )
            cv.circle(
                imfused, end2, circ_radius, (255, 0, 0), line_thickness, cv.LINE_AA
            )

    return imfused
