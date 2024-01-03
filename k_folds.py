import os
import shutil
import glob

# Define paths
path_source = r'D:\Ghazi\Pointcloud'
path_val = r'D:\Ghazi\PointcloudVal1'
path_test = r'D:\Ghazi\PointcloudTest'
def move_files(source, destination, pattern):
    """Move files from source to destination based on a given pattern."""
    print(pattern)
    for file in os.listdir(source):
        # print(file)

        if pattern in file:
            # print('ok')
            try:
                shutil.move(os.path.join(source, file), destination)
            except Exception as e:
                print(f"Error moving file {file}: {e}")


def fold(K):
    """Main function to orchestrate the file moving logic."""
    # Ensure K+1 wraps around to 1 if K is 10
    K_plus_1 = 1 if K == 10 else K + 1

    # Patterns for matching file names
    pattern_I = f'Specimen_{K}'
    pattern_I_plus_1 = f'Specimen_{K_plus_1}'

    # Move files from B and C to A
    for path in [path_val]:#, path_test]:
        move_files(path, path_source, '')

    # Move files based on Specimen patterns
    # print('test')
    move_files(path_source, path_val, pattern_I)
    # move_files(path_source, path_test, pattern_I_plus_1)

if __name__ == "__main__":
    # Get user input
    I = int(input("Enter an integer value for I: "))

    # Run main function
    fold(I)