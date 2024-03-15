import cv2
import numpy as np


def segment_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Create a mask of zeros, with the same dimensions as the image
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define the background and foreground models for GrabCut
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    # Define the rectangle enclosing the object of interest (you might need to adjust this manually)
    rectangle = (50, 50, image.shape[1]-50, image.shape[0]-50)

    # Apply GrabCut algorithm to segment the image
    cv2.grabCut(image, mask, rectangle, background_model,
                foreground_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create a mask where all regions classified as background or probably background are set to 0
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Find contours in the mask to get the bounding box of the object
    contours, _ = cv2.findContours(
        mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to select only the main object
    main_contours = [cnt for cnt in contours if cv2.contourArea(
        cnt) > 1000]  # Adjust the area threshold as needed

    # Find the contour with the largest area
    main_contour = max(
        main_contours, key=cv2.contourArea) if main_contours else None

    if main_contour is not None:
        # Get the bounding box coordinates of the main contour
        x, y, w, h = cv2.boundingRect(main_contour)

        # Draw a bounding box around the main contour on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the area of the bounding box
        box_area = w * h

        # Calculate the percentage of the object visible
        total_area = image.shape[0] * image.shape[1]
        visible_percentage = (box_area / total_area) * 100

        # Display the percentage of visibility above the bounding box
        cv2.putText(image, f"Visibility: {visible_percentage:.2f}%", (
            x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resize the image to fit the screen
    screen_width = 1280  # Set your screen width
    screen_height = 720  # Set your screen height
    ratio = min(screen_width / image.shape[1], screen_height / image.shape[0])
    resized_image = cv2.resize(
        image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))

    return resized_image


# Path to the input image
image_path = 'butter.jpg'

# Perform image segmentation
segmented_image = segment_image(image_path)

# Display the segmented image with bounding box and visibility percentage
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
