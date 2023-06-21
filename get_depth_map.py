import pyzed.sl as sl
import cv2

def main():
    # Create a depth map object
    depth_map = sl.Mat()

    # Load the depth map from the saved dataset
    if depth_map.read("path/to/depth_map.svo") != sl.ERROR_CODE.SUCCESS:
        print("Failed to read the depth map")
        return

    # Get the depth map data as a NumPy array
    depth_map_np = depth_map.get_data()

    # Normalize the depth map for display
    depth_map_np_normalized = cv2.normalize(depth_map_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Display the depth map
    cv2.imshow("Depth Map", depth_map_np_normalized)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()