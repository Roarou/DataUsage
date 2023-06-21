import pyzed.sl as sl
import os

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set the configuration parameters for the camera
    init_params = sl.InitParameters()
    init_params.input.set_from_svo_file("E:/Ghazi/dataset.svo")  # Specify the path to the dataset file

    # Open the camera using the dataset
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open the camera with the dataset")
        return

    # Enable spatial mapping
    mapping_params = sl.SpatialMappingParameters()
    mapping_params.use_chunk_only = True  # Set to True to use only the spatial mapping chunks present in the dataset
    mapping_params.map_type = sl.MAP_TYPE.MESH  # Specify the type of map (can be MESH or FUSED_POINT_CLOUD)

    err = zed.enable_spatial_mapping(mapping_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable spatial mapping")
        zed.close()
        return

    # Start the spatial mapping
    zed.request_spatial_map_async()

    # Process frames until the spatial mapping is complete
    while True:
        if zed.get_spatial_mapping_state() == sl.SpatialMappingState.WORKING:
            # Grab a new frame (not required in this case as we're using a dataset)
            zed.grab()

            # Retrieve the spatial mapping data
            mapping_data = sl.SpatialMappingParameters()

            if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_spatial_map_async(mapping_data)
                break

    # Create the output folder if it doesn't exist
    output_folder = "Spatial_Results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the spatial map to an OBJ file
    output_map_path = os.path.join(output_folder, "output_map.obj")
    zed.save_spatial_map(output_map_path)

    # Close the camera
    zed.close()

    # Open the ZED Depth View tool with the exported spatial map
    zed_depth_view_path = r"C:\Program Files (x86)\ZED SDK\tools\ZED Depth Viewer.exe"  # Path to the ZED Depth Viewer application
    os.system(f'"{zed_depth_view_path}" "{output_map_path}"')

if __name__ == "__main__":
    main()