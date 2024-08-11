import cv2
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import cv2
import matplotlib
#matplotlib.use('Agg')  # Set the backend to non-interactive
from matplotlib import pyplot as plt

def occlusion(image_path):
    font = {'fontsize': 15}  # Example font dictionary
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Red color boundaries [B, G, R]
    lower = [np.mean(image[:, :, channel] - np.std(image[:, :, channel]) / 3) for channel in range(3)]
    upper = [250, 250, 250]
    # Create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    # Find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    # Thresholding
    ret, thresh = cv2.threshold(mask, 40, 255, 0)
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # Draw in blue the contours that were found
        cv2.drawContours(output, contours, -1, 255, 3)
        # Find the biggest contour by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Draw the biggest contour in green
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 5)
        # Extract the region of interest (ROI)
        foreground = image[y:y + h, x:x + w]
        # Create the plot
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 3, 1), plt.imshow(image), plt.title("Input", fontdict=font)
        plt.subplot(1, 3, 2), plt.imshow(output), plt.title("All Contours", fontdict=font)
        plt.subplot(1, 3, 3), plt.imshow(foreground), plt.title("Output", fontdict=font)
        print(foreground.shape)
        
        # Save the plot instead of showing it
        plt.savefig('static/occlusion_result.png')
        plt.close()  # Close the figure to free up memory