import os
import cv2

# Folder path containing the input images
input_folder = "C:/Users/hyeee/Desktop/dataset/DATASET2_B"

# Output folder path to save the rotated images
output_folder = "D:/deeplearning/90"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Rotation angles in degrees
rotation_angles = [45, 90, 135]

# Iterate over each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Get the image dimensions
        height, width = image.shape[:2]

        # Apply rotation for each angle
        for angle in rotation_angles:
            # Calculate the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

            # Apply the rotation to the image
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

            # Create the new filename
            new_filename = filename.replace(".png", f"_rotated{angle}.png")

            # Create the output file path
            output_path = os.path.join(output_folder, new_filename)

            # Save the rotated image with the new filename
            cv2.imwrite(output_path, rotated_image)

            #print(f"Rotated image saved: {output_path}")

print("Image rotation completed.")
