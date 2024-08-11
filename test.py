
import cv2
from ellipseRecognize import DTECMA
import matplotlib.pyplot as plt


def test(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    ellipses, output = DTECMA(binary, lam = 0.75)
    plt.imshow(output)
    plt.savefig('static/output_completion.png')
    plt.close()

# # Read and preprocess the image
# image = plt.imread('occlusion1.png')
# image_filled = fill_shapes(image)

# # Detect ellipses
# ellipses = findEllipse(image_filled)

# # Optionally overlay and plot ellipses
# image_with_ellipses = overlay(image_filled, ellipses)
# image_with_ellipses = plotEllipse(image_with_ellipses, ellipses)

# # Display or save the result
# plt.imshow(image_with_ellipses, cmap='gray')
# plt.show()
