import cv2
import numpy as np

def draw_text_overlay(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.6,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) ->np.ndarray:
    
    """Return a copy of the frane with text drawn on it."""
    
    
    output_frame = frame.copy()
    
    # Draw the text onto the copied frame.
    cv2.putText(
        output_frame,
        text,
        position,
        cv2.FONT_HERSHEY_COMPLEX,
        font_scale,
        color_bgr,  # -> OpenCV use BGR not RGB
        thickness,
    )
    
    
    return output_frame

def resize_to_fit(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    """Shrink the frame to fit inside max_width x max_height, keeping its aspect ratio."""
    height, width = frame.shape[:2]

    # Pick the smaller scale so BOTH sides fit. Never enlarge (cap at 1.0).
    scale = min(max_width / width, max_height / height, 1.0)

    if scale == 1.0:
        return frame

    # cv2.resize takes (width, height), NOT (height, width) like NumPy shapes.
    new_size = (int(width * scale), int(height * scale))

    # INTER_AREA gives the sharpest result when shrinking.
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)