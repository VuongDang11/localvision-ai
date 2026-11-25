import cv2 
import time 
import numpy as np 

from datetime import datetime
from pathlib import Path 


from app.video_source import VideoSource
from app.frame_utils import draw_text_overlay
from app.frame_utils import resize_to_fit

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"

SAMPLE_VIDEO_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "sample_videos" / "driveway.mp4"
)

def save_snapshot(
    frame: np.ndarray,
    directory: Path = SNAPSHOT_DIR
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / "test_snapshot.jpg"

    if not cv2.imwrite(str(path), frame):
        raise RuntimeError("Failed to save snapshot")

    return path

def main() :
    
    """
    This is used for Own Camera
    """
    
    # camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # DSHOW = Direct Show to speeds up the initialization avoid long lag times / freeze 

    # # Fall back to the default camera backend.
    # if not camera.isOpened():
    #     camera.release()
    #     camera = cv2.VideoCapture(0) 

    # if not camera.isOpened():
    #     raise RuntimeError("Cannot open webcam")
    
    
    camera = VideoSource(str(SAMPLE_VIDEO_PATH), loop=True, realtime=True)
    camera.open()

    # WINDOW_NORMAL lets you resize the window (this video is tall).
    cv2.namedWindow("LocalVision AI v0.1", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.perf_counter() # measure how fast frames are being processed 
    measured_fps = 0.0
    printed_info = False

    # try:
    #     while True:
    #         frame = camera.read()

    #         if frame is None:
    #             print("Failed to read a frame")
    #             break
            
    #         # # Mirror the webcam horizontally
    #         # frame = cv2.flip(frame, 1)

    #         # Display metadata once.
    #         if not printed_info:
    #             print("Shape:", frame.shape)
    #             print("Dimensions:", frame.ndim)
    #             print("Data type:", frame.dtype)
    #             print("Memory:", frame.nbytes, "bytes")
    #             printed_info = True

    #         # Calculate measured loop FPS.
    #         frame_count += 1
    #         now_time = time.perf_counter()
    #         elapsed = now_time - start_time

    #         if elapsed >= 1.0:
    #             measured_fps = frame_count / elapsed
    #             frame_count = 0
    #             start_time = now_time

    #         # Create local timestamp.
    #         now = datetime.now().astimezone()
    #         timestamp = now.strftime("%Y-%m-%d %H:%M:%S") # Fetches the exact current date and time  from my clock using Format : Year-Month-Days Hours-Mins-Secs 

    #         frame = draw_text_overlay(
    #             frame,                  # Draws on your live webcam frame
    #             f"FPS: {measured_fps:.1f}", # Displays text like "FPS: 30.2"
    #             (15, 30),               # Position: 15 pixels from the left, 30 pixels down from the top
    #             0.7,                    # Font scale (size)
    #             (0, 255, 0),            # Color: Green (Blue=0, Green=255, Red=0)
    #             2                       # Line thickness of 2 pixels
    #         )

    #         frame = draw_text_overlay(
    #             frame,
    #             timestamp,
    #             (15, 65),
    #             0.6,
    #             (255, 255, 255),
    #             2
    #         )

    #         cv2.imshow("LocalVision AI v0.1", frame)

    #         key = cv2.waitKey(1) & 0xFF

    #         if key == ord("s"):
    #             SNAPSHOT_DIR.mkdir(
    #                 parents=True,
    #                 exist_ok=True
    #             )

    #             filename = now.strftime(
    #                 "%Y%m%d_%H%M%S_%f.jpg"
    #             )
    #             path = SNAPSHOT_DIR / filename

    #             if cv2.imwrite(str(path), frame):
    #                 print("Saved snapshot:", path)
    #             else:
    #                 print("Snapshot save failed")

    #         elif key == ord("q"):
    #             break

    # finally:
    #     camera.release()
    #     cv2.destroyAllWindows()

    
    """
    This is used for Video Dataset 
    """
    try:
        while True:
            # Read ONE frame per loop iteration.
            raw_frame = camera.read()

            if raw_frame is None:
                print("Failed to read a frame")
                break

            # Display metadata once.
            if not printed_info:
                print("Shape:", raw_frame.shape)
                print("Dimensions:", raw_frame.ndim)
                print("Data type:", raw_frame.dtype)
                print("Memory:", raw_frame.nbytes, "bytes")
                printed_info = True

            # Calculate measured loop FPS.
            frame_count += 1
            now_time = time.perf_counter()
            elapsed = now_time - start_time

            if elapsed >= 1.0:
                measured_fps = frame_count / elapsed
                frame_count = 0
                start_time = now_time

            # Create local timestamp.
            now = datetime.now().astimezone()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            # Resize a copy for the screen, then draw text on that copy only.
            display_frame = resize_to_fit(raw_frame, max_width=960, max_height=720)

            display_frame = draw_text_overlay(
                display_frame,
                f"FPS: {measured_fps:.1f}",
                position=(15, 30),
                font_scale=0.7,
                color_bgr=(0, 255, 0),
                thickness=2,
            )

            display_frame = draw_text_overlay(
                display_frame,
                timestamp,
                position=(15, 65),
                font_scale=0.6,
                color_bgr=(255, 255, 255),
                thickness=2,
            )

            cv2.imshow("LocalVision AI v0.1", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S")):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

                filename = now.strftime("%Y%m%d_%H%M%S_%f.jpg")
                path = SNAPSHOT_DIR / filename

                # Save the clean, full-size frame (no text overlay).
                if cv2.imwrite(str(path), raw_frame):
                    print("Saved snapshot:", path)
                else:
                    print("Snapshot save failed")

            elif key in (ord("q"), ord("Q")):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()