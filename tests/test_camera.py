import cv2
import numpy as np 

 
from app.camera import save_snapshot 


def test_saved_image_decision(tmp_path): 
    # Arrange : Create a fake BGR image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Act: Save the image to a temp directory
    saved_path = save_snapshot(frame, tmp_path)
    
    # Read the saved image 
    loaded = cv2.imread(str(saved_path))
    
    # Assert: Check the results
    assert saved_path.is_file()
    assert loaded is not None
    assert loaded.shape == (480,640,3)
    assert loaded.dtype == np.uint8
