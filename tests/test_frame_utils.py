import numpy as np

from app.frame_utils import draw_text_overlay


def test_draw_text_overlay_draws_text_and_keeps_original():
    # Arrange: a fully black image (all pixel values are 0)
    blank_frame = np.zeros((100, 300, 3), dtype=np.uint8)

    # Act: draw white text on it
    frame_with_text = draw_text_overlay(blank_frame, "Hello", (10, 50))

    # Assert 1: the output has the same size as the input
    assert frame_with_text.shape == blank_frame.shape

    # Assert 2: some pixels are no longer black, so text was drawn
    assert frame_with_text.any()

    # Assert 3: the original frame is still fully black (not modified)
    assert not blank_frame.any()


from app.frame_utils import draw_text_overlay, resize_to_fit


def test_resize_to_fit_keeps_aspect_ratio():
    # Arrange: a tall portrait frame like the sample video
    portrait_frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    # Act: fit it into a 960 x 720 display area
    resized_frame = resize_to_fit(portrait_frame, max_width=960, max_height=720)

    # Assert: height hits the limit, width shrinks by the same ratio
    assert resized_frame.shape == (720, 405, 3)


def test_resize_to_fit_does_not_enlarge_small_frames():
    small_frame = np.zeros((100, 200, 3), dtype=np.uint8)

    resized_frame = resize_to_fit(small_frame, max_width=960, max_height=720)

    assert resized_frame.shape == small_frame.shape