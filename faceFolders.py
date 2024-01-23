import matplotlib.pyplot as plt
import face_recognition
import os
import numpy as np

# Define the paths to your folders
folder_a = 'folder_a'  # Replace with your actual folder path for input images
folder_b = 'folder_b'  # Replace with your actual folder path for output images

# Define your threshold
prune_threshold = 0.7  # You can adjust this value as needed

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
    # Get unique input image filenames from the scores
    input_images_filenames = list(set([score[0] for score in scores if score[2] <= prune_threshold]))
    num_input_images = len(input_images_filenames)

    # Sort scores by input image name and score, and only include those below the threshold
    sorted_scores = [score for score in sorted(scores, key=lambda x: (x[0], x[2])) if score[2] <= prune_threshold]

    # Create the figure with dynamic subplot count for up to 10 matches
    # We subtract one because the first column will be used for the input images
    fig, axs = plt.subplots(num_input_images, 10, figsize=(30, num_input_images * 3))

    # Ensure axs is a 2D array even if there's only one input image
    if num_input_images == 1:
        axs = np.array([axs]).reshape(-1, 10)  # Adjusted to ensure 2D array

    # Plot images and scores
    for i, input_img_filename in enumerate(input_images_filenames):
        input_img_path = os.path.join(input_folder, input_img_filename)
        input_img = plt.imread(input_img_path)
        input_scores = [score for score in sorted_scores if score[0] == input_img_filename]

        # Display the input image in the first column
        axs[i, 0].imshow(input_img)
        axs[i, 0].set_title(input_img_filename)
        axs[i, 0].axis('off')

        # Loop over the next columns for potential matches
        for j, score in enumerate(input_scores[:9], start=1):  # Adjusted to start from the second column
            comparison_img_path = os.path.join(comparison_folder, score[1])
            comparison_img = plt.imread(comparison_img_path)
            axs[i, j].imshow(comparison_img)
            axs[i, j].set_title(f"{score[1]}\nScore: {score[2]:.2f}")
            axs[i, j].axis('off')

        # Turn off any unused axes
        for j in range(len(input_scores) + 1, 10):
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('output_grid.png')
    plt.close()

# Call the function to plot the grid
plot_image_grid(scores, folder_a, folder_b)
