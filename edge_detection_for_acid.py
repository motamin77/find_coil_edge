import cv2
import numpy as np

# ==============================================================================
# Adjustable Parameters (Constants)
# You can fine-tune the algorithm by modifying these values.
# ==============================================================================

# Number of columns to crop from the right side of the image to avoid edge artifacts.
CROP_COLUMNS = 5

# Minimum brightness difference between the two regions around an edge to consider it "significant".
MIN_BRIGHTNESS_DIFF_FOR_EDGE = 10

# Percentage of image columns that must be blue-dominant to be classified as a "full slab".
BLUE_DOMINANCE_THRESHOLD_PERCENT = 0.8  # (80%)

# Kernel size for the Sobel edge detection filter (must be an odd number).
SOBEL_KERNEL_SIZE = 5

# Window size for sampling brightness near the detected edge.
BRIGHTNESS_WINDOW_SIZE = 20

# Offset from the detected edge to start the sampling window.
BRIGHTNESS_WINDOW_OFFSET = 10




def _calculate_edge_profile(gray_image):
    """
    A helper function to calculate the edge strength profile using the Sobel filter.
    """
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)
    
    # Calculate the magnitude of the gradient
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Calculate the mean edge strength for each column
    return np.mean(edge_magnitude, axis=0)


def analyze_slab_image(input_image, camera_direction):
    """
    Analyzes the input image to determine the slab's status.

    Args:
        input_image (np.ndarray): The input image as a NumPy array (read by OpenCV).
        camera_direction (str): The camera's orientation. Valid values: "Top Camera" or "Bottom Camera".

    Returns:
        int:
            - A positive integer (X): The pixel column number where the edge is located.
            - 1: If the image is determined to be a full slab.
            - -1: If no slab is detected.
    """
    if input_image is None or input_image.size == 0:
        raise ValueError("Input image is invalid.")

    # --- Step 1: Select the Region of Interest (ROI) ---
    height, _, _ = input_image.shape
    if camera_direction == "Top Camera":
        # Lower half of the image
        start_row = height // 2
        end_row = height
    elif camera_direction == "Bottom Camera":
        # Upper half of the image
        start_row = 0
        end_row = height // 2
    else:
        raise ValueError("Invalid camera_direction. Must be 'Top Camera' or 'Bottom Camera'.")
    
    roi_image = input_image[start_row:end_row, :]
    
    if roi_image.size == 0:
        return -1 # If ROI is empty, it means no slab is present

    # --- Step 2: Preprocessing ---
    # Crop columns from the right side
    analysis_image = roi_image[:, :-CROP_COLUMNS]
    
    # Convert to grayscale for brightness and edge analysis
    gray_image = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2GRAY)

    # --- Step 3: Calculate Image Profiles ---
    mean_intensity_profile = np.mean(gray_image, axis=0)
    edge_strength_profile = _calculate_edge_profile(gray_image)

    # --- Step 4: Find the Potential Edge Position ---
    # The edge is located at the column with the highest edge strength
    edge_position = np.argmax(edge_strength_profile)

    # --- Step 5: Evaluate Contrast Around the Edge ---
    width = analysis_image.shape[1]
    
    # Define sampling windows on both sides of the edge
    start_1 = max(0, edge_position - BRIGHTNESS_WINDOW_SIZE - BRIGHTNESS_WINDOW_OFFSET)
    end_1 = max(0, edge_position - BRIGHTNESS_WINDOW_OFFSET)
    start_2 = min(width, edge_position + BRIGHTNESS_WINDOW_OFFSET)
    end_2 = min(width, edge_position + BRIGHTNESS_WINDOW_SIZE + BRIGHTNESS_WINDOW_OFFSET)
    
    brightness_difference = 0
    if end_1 > start_1 and end_2 > start_2:
        zone1 = mean_intensity_profile[start_1:end_1]
        zone2 = mean_intensity_profile[start_2:end_2]

        if zone1.size > 0 and zone2.size > 0:
            mean1, mean2 = np.mean(zone1), np.mean(zone2)
            brightness_difference = abs(mean1 - mean2)

    # --- Step 6: Final Decision and Output Generation ---
    
    # 6.1: Check for a significant edge
    if brightness_difference > MIN_BRIGHTNESS_DIFF_FOR_EDGE:
        # A strong edge was detected
        return int(edge_position)
    
    # 6.2: If no edge, use color analysis to detect full slab vs. no slab
    else:
        # Split the color channels from the ROI image
        b, g, r = cv2.split(analysis_image)
        mean_b_profile = np.mean(b, axis=0)
        mean_g_profile = np.mean(g, axis=0)
        mean_r_profile = np.mean(r, axis=0)
        
        # Count the columns where blue is the dominant color
        blue_dominant_cols = np.sum((mean_b_profile > mean_g_profile) & (mean_b_profile > mean_r_profile))
        total_cols = analysis_image.shape[1]
        
        blue_percentage = blue_dominant_cols / total_cols
        
        if blue_percentage > BLUE_DOMINANCE_THRESHOLD_PERCENT:
            # Full slab detected
            return 1
        else:
            # No slab detected
            return -1

# ==============================================================================
# Example and Test Section
# To run this section, place an image named 'sample_image.jpg' in the same directory.
# ==============================================================================
if __name__ == "__main__":
    try:
        # Load a sample image.
        # Replace 'path/to/your/image.jpg' with the actual path to your image.
        sample_image = cv2.imread("sample_image.jpg")

        if sample_image is None:
            print("Error: The sample image file was not found or could not be read.")
            print("Please place a valid image at the specified path.")
        else:
            print("--- Testing Algorithm with Sample Image ---")
            
            # Test assuming it's the top camera
            camera_pos_top = "Top Camera"
            result_top = analyze_slab_image(sample_image, camera_pos_top)
            
            print(f"\nResult for '{camera_pos_top}': {result_top}")
            if result_top > 1:
                print(f"-> Status: Edge detected at column {result_top}.")
            elif result_top == 1:
                print("-> Status: Full Slab detected.")
            elif result_top == -1:
                print("-> Status: No Slab detected.")

            print("-" * 20)

            # Test assuming it's the bottom camera
            camera_pos_bottom = "Bottom Camera"
            result_bottom = analyze_slab_image(sample_image, camera_pos_bottom)

            print(f"Result for '{camera_pos_bottom}': {result_bottom}")
            if result_bottom > 1:
                print(f"-> Status: Edge detected at column {result_bottom}.")
            elif result_bottom == 1:
                print("-> Status: Full Slab detected.")
            elif result_bottom == -1:
                print("-> Status: No Slab detected.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

### How to Use:

# 1.  Save the code above into a file named `slab_analyzer.py`.
# 2.  Make sure you have the necessary libraries installed:
#     ```bash
#     pip install opencv-python numpy
#     ```
# 3.  To test it, place a sample image in the same directory and rename it to `sample_image.jpg`. Then, run the script from your terminal:
#     ```bash
#     python slab_analyzer.py
#     ```
# 4.  To integrate it into your main application, import the function and call it with your image data:

#     ```python
#     # In your main application file
#     import cv2
#     from slab_analyzer import analyze_slab_image

#     # Read a frame from your camera or a file
#     my_image_frame = cv2.imread("path/to/an/image.png")
#     camera_id = 1
#     camera_orientation = "Top Camera" # Or "Bottom Camera"

#     # Call the analysis function
#     analysis_result = analyze_slab_image(my_image_frame, camera_orientation)

#     # Make decisions based on the result
#     print(f"Analysis result for camera {camera_id}: {analysis_result}")
    