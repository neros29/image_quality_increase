import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models

# Function to load the trained super-resolution model
def load_super_resolution_model(model_path='super_resolution_model.h5'):
    try:
        model = models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to apply super-resolution model to a folder of images
def enhance_images_in_folder(input_folder, output_folder, model, target_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        
        # Skip directories or non-image files
        if os.path.isdir(img_path) or not filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            continue
        
        try:
            # Load and preprocess the image
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)  # Resize to the model's input size (default 256x256)
            img_array = np.array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict enhanced image using the model
            enhanced_image = model.predict(img_array)
            enhanced_image = np.squeeze(enhanced_image, axis=0)  # Remove batch dimension
            enhanced_image = np.clip(enhanced_image * 255.0, 0, 255).astype(np.uint8)  # De-normalize and convert to uint8

            # Save the enhanced image to the output folder
            output_img = Image.fromarray(enhanced_image)
            output_path = os.path.join(output_folder, filename)
            output_img.save(output_path)

            print(f"Enhanced {filename} and saved to {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Function to display enhanced images for comparison
def display_image_comparison(input_folder, output_folder, num_images=3):
    # Display a few images before and after enhancement
    input_images = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    output_images = [f for f in os.listdir(output_folder) if f in input_images]  # Match input and output filenames

    plt.figure(figsize=(15, 5 * num_images))
    for i, filename in enumerate(input_images[:num_images]):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load and display input image
        input_img = Image.open(input_path)
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(input_img)
        plt.title(f"Original Image: {filename}")
        plt.axis('off')

        # Load and display output image
        if os.path.exists(output_path):
            output_img = Image.open(output_path)
            plt.subplot(num_images, 2, 2 * i + 2)
            plt.imshow(output_img)
            plt.title(f"Enhanced Image: {filename}")
            plt.axis('off')

    plt.show()

# Main function to run the enhancement process
def main():
    # Set paths for input and output folders
    input_folder = '/content/low_res_images'  # Replace with the path to your folder containing low-res images
    output_folder = '/content/enhanced_images'  # Replace with desired output folder path

    # Load the pre-trained super-resolution model
    model = load_super_resolution_model('super_resolution_model.h5')  # Ensure model is in the correct path

    if model is not None:
        # Specify the target size for enhancement
        target_size = (256, 256)  # Replace with desired target size or use a scaling factor

        # Enhance images in the input folder
        enhance_images_in_folder(input_folder, output_folder, model, target_size=target_size)

        # Display a few images for comparison
        display_image_comparison(input_folder, output_folder)

# Run the main function if script is executed
if __name__ == "__main__":
    main()
