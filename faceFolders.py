import matplotlib.pyplot as plt
import face_recognition
import os
import numpy as np

# Define the paths to your folders
folder_a = 'folder_a'  # Replace with your actual folder path for input images
folder_b = 'folder_b'  # Replace with your actual folder path for output images

# Function to load images and calculate scores
def load_and_score_images(input_folder, comparison_folder):
    input_images = {}
    comparison_images = {}
    scores = []

    # Load input images and compute face encodings
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        image = face_recognition.load_image_file(img_path)
        input_images[img_name] = face_recognition.face_encodings(image)[0]

    # Load comparison images and compute face encodings
    for img_name in os.listdir(comparison_folder):
        img_path = os.path.join(comparison_folder, img_name)
        image = face_recognition.load_image_file(img_path)
        comparison_images[img_name] = face_recognition.face_encodings(image)[0]

    # Compare each input image with each comparison image and store scores
    for input_name, input_encoding in input_images.items():
        for comparison_name, comparison_encoding in comparison_images.items():
            score = face_recognition.face_distance([input_encoding], comparison_encoding)
            scores.append((input_name, comparison_name, score[0]))
    
    return scores


# Call the above function and get the scores
scores = load_and_score_images(folder_a, folder_b)

# Function to plot the image grid
def plot_image_grid(scores, input_folder, comparison_folder):
    # List all image filenames in input_folder
    input_images_filenames = [img for img in os.listdir(input_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_input_images = len(input_images_filenames)

    # Sort scores by input image name and score
    sorted_scores = sorted(scores, key=lambda x: (x[0], x[2]))

    # Initialize a plot grid
    fig, axs = plt.subplots(num_input_images, 11, figsize=(15, num_input_images * 3))

    # Ensure axs is a 2D array even if there's only one input image
    if num_input_images == 1:
        axs = np.array([axs])

    # Plot images and scores
    for input_img_filename in input_images_filenames:
        input_img_path = os.path.join(input_folder, input_img_filename)
        input_img = plt.imread(input_img_path)
        input_scores = [score for score in sorted_scores if score[0] == input_img_filename]
        
        for idx, (input_name, comparison_name, score) in enumerate(input_scores[:10]):  # Get top 10 scores for this input
            comparison_img_path = os.path.join(comparison_folder, comparison_name)
            comparison_img = plt.imread(comparison_img_path)

            axs[input_images_filenames.index(input_img_filename), idx].imshow(comparison_img if idx > 0 else input_img)
            axs[input_images_filenames.index(input_img_filename), idx].set_title(f"{comparison_name}\nScore: {score:.2f}" if idx > 0 else input_img_filename)
            axs[input_images_filenames.index(input_img_filename), idx].axis('off')

    plt.tight_layout()
    plt.savefig('output_grid.png')
    plt.show()

# Call the function to plot the grid
plot_image_grid(scores, folder_a, folder_b)
