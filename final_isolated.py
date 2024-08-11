# import numpy as np
# import cv2
# from scipy.interpolate import splprep, splev
# from matplotlib import pyplot as plt

# def smooth_contour(contour, epsilon=0.01):
#     perimeter = cv2.arcLength(contour, True)
#     return cv2.approxPolyDP(contour, epsilon * perimeter, True)

# def fit_spline(points, num_points=100):
#     if len(points) < 3:
#         return points  # Not enough points to fit a spline

#     x, y = points.T
#     t = np.linspace(0, 1, len(x))

#     k = min(3, len(x) - 1)  # Adjust the degree based on the number of points
    
#     tck, u = splprep([x, y], s=0, k=k)
#     new_points = splev(np.linspace(0, 1, num_points), tck)
#     return np.column_stack(new_points).astype(np.int32)

# def find_shapes(image_bgr):
#     image_output = np.ones_like(image_bgr) * 255  # Create a white background
#     image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     for contour in contours:
#         if cv2.arcLength(contour, True) <= 100:
#             continue
        
#         smoothed_contour = smooth_contour(contour)
#         smoothed_contour = smoothed_contour.reshape(-1, 2)  # Flatten contour for spline fitting
        
#         spline_points = fit_spline(smoothed_contour)
        
#         # Draw the smoothed contour with a unique color
#         cv2.polylines(image_output, [spline_points], isClosed=True, color=(0, 255, 0), thickness=2)

#     return image_output

# def show_output(input_img, output_img):
#     plt.figure(figsize=(12, 6))

#     plt.subplot(121)
#     plt.axis('off')
#     plt.title("Input")
#     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(122)
#     plt.axis('off')
#     plt.title("Output")
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))

#     plt.show()

# # Main code to process a single image file
# def process_single_image(image_path):
#     image = cv2.imread(image_path)
#     output_image = find_shapes(image)
#     show_output(image, output_image)

# # Example usage
# image_path = "images/isolated.png"  # Replace with your image path
# process_single_image(image_path)


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import measure

# # Load the image
# image_path = 'images/isolated.png'  # Path to your image
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Threshold the image to binary
# _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# # Find contours
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Initialize the plot
# plt.figure(figsize=(10, 10))

# # Process each contour
# for contour in contours:
#     # Get the approximated shape for each contour
#     epsilon = 0.01 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
    
#     # Fit an ellipse or polygon based on the number of points in the contour
#     if len(approx) > 8:  # This is likely a curve or circle
#         ellipse = cv2.fitEllipse(contour)
#         cv2.ellipse(image, ellipse, (0, 255, 0), 2)
#         label = 'Ellipse'
#     else:  # This is likely a polygon
#         cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)
#         label = f'Polygon with {len(approx)} sides'

#     # Calculate the center of the contour for labeling
#     M = cv2.moments(contour)
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#     else:
#         cX, cY = contour[0][0]
    
#     # Annotate the contour with its label
#     plt.text(cX, cY, label, fontsize=12, color='white', ha='center')

# # Display the final image with labeled shapes
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image_path = 'images/isolated.png'  # Path to your image
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Threshold the image to binary
# _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# # Find contours
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Initialize the plot
# plt.figure(figsize=(10, 10))

# # Process each contour
# for contour in contours:
#     # Fit an ellipse if the shape is complex, otherwise approximate it
#     if len(contour) > 8:
#         ellipse = cv2.fitEllipse(contour)
#         center, axes, angle = ellipse
#         ellipse_curve = cv2.ellipse2Poly((int(center[0]), int(center[1])), (int(axes[0] / 2), int(axes[1] / 2)), int(angle), 0, 360, 1)
#         plt.plot(ellipse_curve[:, 0], ellipse_curve[:, 1], 'g-', linewidth=2)
#     else:
#         epsilon = 0.01 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         approx = np.squeeze(approx)  # Remove unnecessary dimensions
#         plt.plot(approx[:, 0], approx[:, 1], 'b-', linewidth=2)

# # Set the aspect ratio to be equal and remove axis
# plt.gca().set_aspect('equal', adjustable='box')
# plt.axis('off')

# # Display the final smoothed curves
# plt.show()


# import numpy as np
# import cv2
# from scipy.interpolate import splprep, splev
# from matplotlib import pyplot as plt

# def smooth_contour(contour, epsilon=0.01):
#     perimeter = cv2.arcLength(contour, True)
#     return cv2.approxPolyDP(contour, epsilon * perimeter, True)

# def fit_spline(points, num_points=100):
#     if len(points) < 3:  # Ensure there are enough points for spline fitting
#         return points
    
#     x, y = points.T
#     k = min(3, len(points) - 1)  # Adjust spline degree based on the number of points
#     tck, u = splprep([x, y], s=0, k=k)
#     new_points = splev(np.linspace(0, 1, num_points), tck)
#     return np.column_stack(new_points).astype(np.int32)

# def find_shapes(image_bgr):
#     image_output = np.ones_like(image_bgr) * 255  # White background
#     image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     for contour in contours:
#         if cv2.arcLength(contour, True) <= 100:
#             continue
        
#         smoothed_contour = smooth_contour(contour)
#         smoothed_contour = smoothed_contour.reshape(-1, 2)  # Flatten contour for spline fitting
        
#         spline_points = fit_spline(smoothed_contour)
        
#         # Draw the smoothed contour with a unique color
#         cv2.polylines(image_output, [spline_points], isClosed=True, color=(0, 255, 0), thickness=2)

#     return image_output

# def show_output(input_img, output_img):
#     plt.figure(figsize=(12, 6))

#     plt.subplot(121)
#     plt.axis('off')
#     plt.title("Input")
#     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(122)
#     plt.axis('off')
#     plt.title("Output")
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))

#     plt.show()

# # Main code to process a single image file
# def process_single_image(image_path):
#     image = cv2.imread(image_path)
#     output_image = find_shapes(image)
#     show_output(image, output_image)

# # Example usage
# image_path = "images/isolated.png"  # Replace with your image path
# process_single_image(image_path)


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_shape(contour):
#     """Function to detect the shape of a contour."""
#     # Approximate the contour
#     epsilon = 0.04 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#     shape = "unidentified"

#     # Detect based on the number of vertices
#     if len(approx) == 3:
#         shape = "triangle"
#     elif len(approx) == 4:
#         # Check for square or rectangle
#         (x, y, w, h) = cv2.boundingRect(approx)
#         ar = w / float(h)
#         shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
#     elif len(approx) > 4:
#         shape = "circle"
    
#     return shape, approx

# def replace_with_shape(approx, shape):
#     """Replace detected shape with its ideal form."""
#     if shape == "triangle":
#         ret, triangle = cv2.minEnclosingTriangle(approx)
#         return triangle.reshape(-1, 2)
#     elif shape == "square":
#         (x, y, w, h) = cv2.boundingRect(approx)
#         return np.array([
#             [x, y],
#             [x + w, y],
#             [x + w, y + h],
#             [x, y + h]
#         ])
#     elif shape == "circle":
#         center, radius = cv2.minEnclosingCircle(approx)
#         circle = []
#         for angle in np.linspace(0, 2 * np.pi, num=100):
#             circle.append([
#                 int(center[0] + radius * np.cos(angle)),
#                 int(center[1] + radius * np.sin(angle))
#             ])
#         return np.array(circle)
#     else:
#         return approx.reshape(-1, 2)

# # Load the image
# image_path = 'images/isolated.png'  # Path to your image
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Threshold the image to binary
# _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# # Find contours
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Initialize the plot
# plt.figure(figsize=(10, 10))

# # Process each contour
# for contour in contours:
#     # Detect the shape of the contour
#     shape, approx = detect_shape(contour)

#     # Replace the contour with the detected shape's ideal form
#     ideal_shape = replace_with_shape(approx, shape)

#     # Plot the ideal shape if it's not empty
#     if ideal_shape.shape[1] == 2:
#         plt.plot(ideal_shape[:, 0], ideal_shape[:, 1], 'g-', linewidth=2)

# # Set the aspect ratio to be equal and remove axis
# plt.gca().set_aspect('equal', adjustable='box')
# plt.axis('off')

# # Display the final replaced shapes
# plt.show()


# import numpy as np
# from matplotlib import pyplot as plt
# import cv2
# from functions import reduce_noise_median, reduce_noise_morph

# def detect_shape_and_approx(contour):
#     """Function to detect the shape and get the approximated contour."""
#     shape = "unidentified"
#     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

#     if len(approx) == 3:
#         shape = "triangle"
#     elif len(approx) == 4:
#         x, y, w, h = cv2.boundingRect(approx)
#         ratio = float(w)/h
#         if ratio >= 0.95 and ratio <= 1.05:
#             shape = "square"
#         else:
#             shape = "rectangle"
#     elif len(approx) == 5:
#         shape = "pentagon"
#     elif len(approx) == 6:
#         shape = "hexagon"
#     elif len(approx) > 6 and len(approx) <= 10:
#         shape = "star"
#     else:
#         shape = "ellipse"
    
#     return shape, approx

# def replace_with_shape(approx, shape):
#     """Replace detected shape with its ideal form."""
#     if shape == "triangle":
#         ret, triangle = cv2.minEnclosingTriangle(approx)
#         return triangle.reshape(-1, 2)
#     elif shape == "square" or shape == "rectangle":
#         (x, y, w, h) = cv2.boundingRect(approx)
#         return np.array([
#             [x, y],
#             [x + w, y],
#             [x + w, y + h],
#             [x, y + h]
#         ])
#     elif shape == "pentagon" or shape == "hexagon":
#         center, radius = cv2.minEnclosingCircle(approx)
#         num_sides = 5 if shape == "pentagon" else 6
#         poly = []
#         for angle in np.linspace(0, 2 * np.pi, num=num_sides, endpoint=False):
#             poly.append([
#                 int(center[0] + radius * np.cos(angle)),
#                 int(center[1] + radius * np.sin(angle))
#             ])
#         return np.array(poly)
#     elif shape == "star":
#         # Placeholder for star (return approx for now)
#         return approx.reshape(-1, 2)
#     elif shape == "ellipse":
#         ellipse = cv2.fitEllipse(approx)
#         ellipse_points = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
#         return np.array(ellipse_points)
#     else:
#         return approx.reshape(-1, 2)

# def find_shapes(image_bgr):
#     image_output = image_bgr.copy()
#     width = image_bgr.shape[1]
#     image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
#     _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     lines = []

#     image_edges = cv2.Canny(image_binary, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(image_edges, 1, np.pi/180, 200)

#     plt.figure(figsize=(10, 10))

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             ideal_shape = np.array([[x1, y1], [x2, y2]])
#             plt.plot(ideal_shape[:, 0], ideal_shape[:, 1], 'r-', linewidth=2)

#     for contour in contours:
#         if (cv2.arcLength(contour, True) <= 100):
#             continue

#         # Detect the shape and approximate contour
#         shape, approx = detect_shape_and_approx(contour)

#         # Replace the contour with the ideal shape
#         ideal_shape = replace_with_shape(approx, shape)

#         # Plot the ideal shape
#         plt.plot(ideal_shape[:, 0], ideal_shape[:, 1], 'g-', linewidth=2)

#     # Set the aspect ratio to be equal and remove axis
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.axis('off')

#     # Display the final replaced shapes
#     plt.show()

# def show_regular(input_img, grey_img, bin_img, output_img, title="figure1"):
#     fig, _ = plt.subplots(figsize=(10, 10))
#     fig.canvas.setWindowTitle(title)

#     plt.subplot(221)
#     plt.axis('off')
#     plt.title("input")
#     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(222)
#     plt.axis('off')
#     plt.title("greyscale")
#     plt.imshow(cv2.cvtColor(grey_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(223)
#     plt.axis('off')
#     plt.title("binary")
#     plt.imshow(cv2.cvtColor(bin_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(224)
#     plt.axis('off')
#     plt.title("output")
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
#     plt.show()

#     return

# def calculate_noise_level(image):
#     noise_level = np.std(image)
#     return noise_level

# def apply_noise_reduction(image):
#     noise_level = calculate_noise_level(image)

#     if noise_level < 10:
#         # Apply low noise reduction function
#         reduced_noise_image = image
#     elif noise_level < 50:
#         # Apply regular noise reduction function
#         reduced_noise_image = reduce_noise_median(image)
#     else:
#         reduced_noise_image = reduce_noise_median(image)
#         reduced_noise_image = reduce_noise_morph(image)
    
#     return reduced_noise_image
    

# # Main code to process a single image file
# def process_single_image(image_path):
#     image = cv2.imread(image_path)
#     reduced_noise_image = apply_noise_reduction(image)
#     output_image, grey_img, bin_img = find_shapes(reduced_noise_image)
#     show_regular(reduced_noise_image, grey_img, bin_img, output_image, title="Shape Detection")

# # Example usage
# image_path = "images/isolated.png"  # Replace with your image path
# process_single_image(image_path)


# import numpy as np
# from matplotlib import pyplot as plt
# import cv2
# from functions import reduce_noise_median, reduce_noise_morph

# def detect_shape_and_approx(contour):
#     """Function to detect the shape and get the approximated contour."""
#     shape = "unidentified"
#     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

#     if len(approx) == 3:
#         shape = "triangle"
#     elif len(approx) == 4:
#         x, y, w, h = cv2.boundingRect(approx)
#         ratio = float(w)/h
#         if ratio >= 0.95 and ratio <= 1.05:
#             shape = "square"
#         else:
#             shape = "rectangle"
#     elif len(approx) == 5:
#         shape = "pentagon"
#     elif len(approx) == 6:
#         shape = "hexagon"
#     elif len(approx) > 6 and len(approx) <= 10:
#         shape = "star"
#     else:
#         shape = "ellipse"
    
#     return shape, approx

# def replace_with_shape(approx, shape):
#     """Replace detected shape with its ideal form."""
#     if shape == "triangle":
#         ret, triangle = cv2.minEnclosingTriangle(approx)
#         return triangle.reshape(-1, 2)
#     elif shape == "square" or shape == "rectangle":
#         (x, y, w, h) = cv2.boundingRect(approx)
#         return np.array([
#             [x, y],
#             [x + w, y],
#             [x + w, y + h],
#             [x, y + h]
#         ])
#     elif shape == "pentagon" or shape == "hexagon":
#         center, radius = cv2.minEnclosingCircle(approx)
#         num_sides = 5 if shape == "pentagon" else 6
#         poly = []
#         for angle in np.linspace(0, 2 * np.pi, num=num_sides, endpoint=False):
#             poly.append([
#                 int(center[0] + radius * np.cos(angle)),
#                 int(center[1] + radius * np.sin(angle))
#             ])
#         return np.array(poly)
#     elif shape == "star":
#         # Placeholder for star (return approx for now)
#         return approx.reshape(-1, 2)
#     elif shape == "ellipse":
#         ellipse = cv2.fitEllipse(approx)
#         ellipse_points = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
#         return np.array(ellipse_points)
#     else:
#         return approx.reshape(-1, 2)

# def find_shapes(image_bgr):
#     image_output = image_bgr.copy()
#     width = image_bgr.shape[1]
#     image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
#     _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)

#     # Morphological operations to close gaps and merge boundaries
#     kernel = np.ones((5, 5), np.uint8)
#     image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel)
    
#     contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     plt.figure(figsize=(10, 10))

#     for contour in contours:
#         if (cv2.arcLength(contour, True) <= 100):
#             continue

#         # Detect the shape and approximate contour
#         shape, approx = detect_shape_and_approx(contour)

#         # Replace the contour with the ideal shape
#         ideal_shape = replace_with_shape(approx, shape)

#         # Plot the ideal shape
#         plt.plot(ideal_shape[:, 0], ideal_shape[:, 1], 'g-', linewidth=2)

#     # Set the aspect ratio to be equal and remove axis
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.axis('off')

#     # Display the final replaced shapes
#     plt.show()

# def show_regular(input_img, grey_img, bin_img, output_img, title="figure1"):
#     fig, _ = plt.subplots(figsize=(10, 10))
#     fig.canvas.setWindowTitle(title)

#     plt.subplot(221)
#     plt.axis('off')
#     plt.title("input")
#     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(222)
#     plt.axis('off')
#     plt.title("greyscale")
#     plt.imshow(cv2.cvtColor(grey_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(223)
#     plt.axis('off')
#     plt.title("binary")
#     plt.imshow(cv2.cvtColor(bin_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(224)
#     plt.axis('off')
#     plt.title("output")
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
#     plt.show()

#     return

# def calculate_noise_level(image):
#     noise_level = np.std(image)
#     return noise_level

# def apply_noise_reduction(image):
#     noise_level = calculate_noise_level(image)

#     if noise_level < 10:
#         # Apply low noise reduction function
#         reduced_noise_image = image
#     elif noise_level < 50:
#         # Apply regular noise reduction function
#         reduced_noise_image = reduce_noise_median(image)
#     else:
#         reduced_noise_image = reduce_noise_median(image)
#         reduced_noise_image = reduce_noise_morph(image)
    
#     return reduced_noise_image
    

# # Main code to process a single image file
# def process_single_image(image_path):
#     image = cv2.imread(image_path)
#     reduced_noise_image = apply_noise_reduction(image)
#     output_image, grey_img, bin_img = find_shapes(reduced_noise_image)
#     show_regular(reduced_noise_image, grey_img, bin_img, output_image, title="Shape Detection")

# # Example usage
# image_path = "images/isolated.png"  # Replace with your image path
# process_single_image(image_path)

# import numpy as np
# from matplotlib import pyplot as plt
# import cv2
# from functions import reduce_noise_median, reduce_noise_morph

# def detect_shape_and_approx(contour):
#     """Function to detect the shape and get the approximated contour."""
#     shape = "unidentified"
#     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

#     if len(approx) == 3:
#         shape = "triangle"
#         print("triangle")
#     elif len(approx) == 4:
#         x, y, w, h = cv2.boundingRect(approx)
#         ratio = float(w)/h
#         if ratio >= 0.95 and ratio <= 1.05:
#             shape = "square"
#             print("square")
#         else:
#             shape = "rectangle"
#             print("rectangle")
#     elif len(approx) == 5:
#         shape = "rectangle"
#         print("pentagon/rectangle")
#     elif len(approx) == 6:
#         shape = "hexagon"
#         print("hexagon")
#     elif len(approx) > 6 and len(approx) <= 10:
#         shape = "star"
#         print("star")
#     else:
#         shape = "ellipse"
#         print("ellipse")
    
#     return shape, approx

# def replace_with_shape(approx, shape):
#     """Replace detected shape with its ideal form."""
#     if shape == "triangle":
#         ret, triangle = cv2.minEnclosingTriangle(approx)
#         return triangle.reshape(-1, 2).astype(int)
#     # elif shape == "square" or shape == "rectangle":
#     #     (x, y, w, h) = cv2.boundingRect(approx)
#     #     return np.array([
#     #         [x, y],
#     #         [x + w, y],
#     #         [x + w, y + h],
#     #         [x, y + h]
#     #     ], dtype=int)
    
#     elif shape == "square" or shape == "rectangle":
#         rect = cv2.minAreaRect(approx)
#         box = cv2.boxPoints(rect)
#         box = np.intp(box)
#         return box
#     elif shape == "pentagon" or shape == "hexagon":
#         center, radius = cv2.minEnclosingCircle(approx)
#         num_sides = 5 if shape == "pentagon" else 6
#         poly = []
#         for angle in np.linspace(0, 2 * np.pi, num=num_sides, endpoint=False):
#             poly.append([
#                 int(center[0] + radius * np.cos(angle)),
#                 int(center[1] + radius * np.sin(angle))
#             ])
#         return np.array(poly, dtype=int)
#     elif shape == "star":
#         # Improved star shape generation
#         center, radius = cv2.minEnclosingCircle(approx)
#         outer_points = []
#         inner_points = []
#         for i in range(5):
#             outer_angle = i * 2 * np.pi / 5 - np.pi / 2
#             inner_angle = outer_angle + np.pi / 5
            
#             outer_x = int(center[0] + radius * np.cos(outer_angle))
#             outer_y = int(center[1] + radius * np.sin(outer_angle))
#             outer_points.append([outer_x, outer_y])
            
#             inner_x = int(center[0] + 0.4 * radius * np.cos(inner_angle))
#             inner_y = int(center[1] + 0.4 * radius * np.sin(inner_angle))
#             inner_points.append([inner_x, inner_y])
        
#         star_points = []
#         for i in range(5):
#             star_points.append(outer_points[i])
#             star_points.append(inner_points[i])
        
#         return np.array(star_points, dtype=int)
#     elif shape == "ellipse":
#         ellipse = cv2.fitEllipse(approx)
#         ellipse_points = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), 
#                                           (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), 
#                                           int(ellipse[2]), 0, 360, 5)
#         return np.array(ellipse_points, dtype=int)
#     else:
#         return approx.reshape(-1, 2).astype(int)

# # def find_shapes(image_bgr):
# #     image_output = image_bgr.copy()
# #     width = image_bgr.shape[1]
# #     image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
# #     _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)

# #     # Morphological operations to close gaps and merge boundaries
# #     kernel = np.ones((8, 3), np.uint8)
# #     image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel)
    
# #     contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# #     plt.figure(figsize=(10, 10))

# #     for contour in contours:
# #         if (cv2.arcLength(contour, True) <= 100):
# #             continue

# #         # Detect the shape and approximate contour
# #         shape, approx = detect_shape_and_approx(contour)

# #         # Replace the contour with the ideal shape
# #         ideal_shape = replace_with_shape(approx, shape)

# #         # Plot the ideal shape
# #         plt.plot(ideal_shape[:, 0], ideal_shape[:, 1], 'g-', linewidth=2)

# #     # Set the aspect ratio to be equal and remove axis
# #     plt.gca().set_aspect('equal', adjustable='box')
# #     plt.axis('off')

# #     # Display the final replaced shapes
# #     plt.show()

# #     return image_output, image_gray, image_binary  # Ensure all necessary values are returned

# def find_shapes(image_bgr):
#     image_output = image_bgr.copy()
#     image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
    
#     # Remove the binary thresholding step
#     # _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)

#     # Apply edge detection instead of using binary image
#     edges = cv2.Canny(image_gray, 50, 150)

#     # Optionally, you can still apply morphological operations to the edge image
#     kernel = np.ones((3, 3), np.uint8)
#     edges = cv2.dilate(edges, kernel, iterations=1)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     plt.figure(figsize=(10, 10))

#     for contour in contours:
#         if cv2.arcLength(contour, True) <= 100:
#             continue

#         # Detect the shape and approximate contour
#         shape, approx = detect_shape_and_approx(contour)

#         # Replace the contour with the ideal shape
#         ideal_shape = replace_with_shape(approx, shape)

#         # Plot the ideal shape
#         plt.plot(np.append(ideal_shape[:, 0], ideal_shape[0, 0]),np.append(ideal_shape[:, 1], ideal_shape[0, 1]), 'g-', linewidth=2)

#     # Set the aspect ratio to be equal and remove axis
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.axis('off')

#     # Display the final replaced shapes
#     plt.show()

#     return image_output, image_gray, edges  # Return edges instead of binary image


# # def show_regular(input_img, grey_img, bin_img, output_img, title="figure1"):
# #     fig, _ = plt.subplots(figsize=(10, 10))
# #     fig.canvas.setWindowTitle(title)

# #     plt.subplot(221)
# #     plt.axis('off')
# #     plt.title("input")
# #     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

# #     plt.subplot(222)
# #     plt.axis('off')
# #     plt.title("greyscale")
# #     plt.imshow(cv2.cvtColor(grey_img, cv2.COLOR_BGR2RGB))

# #     plt.subplot(223)
# #     plt.axis('off')
# #     plt.title("binary")
# #     plt.imshow(cv2.cvtColor(bin_img, cv2.COLOR_BGR2RGB))

# #     plt.subplot(224)
# #     plt.axis('off')
# #     plt.title("output")
# #     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
# #     plt.show()

# #     return

# def show_regular(input_img, grey_img, edge_img, output_img, title="figure1"):
#     fig, _ = plt.subplots(figsize=(10, 10))
#     fig.canvas.setWindowTitle(title)

#     plt.subplot(221)
#     plt.axis('off')
#     plt.title("input")
#     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(222)
#     plt.axis('off')
#     plt.title("greyscale")
#     plt.imshow(grey_img, cmap='gray')

#     plt.subplot(223)
#     plt.axis('off')
#     plt.title("edges")
#     plt.imshow(edge_img, cmap='gray')

#     plt.subplot(224)
#     plt.axis('off')
#     plt.title("output")
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
#     plt.show()

#     return

# def calculate_noise_level(image):
#     noise_level = np.std(image)
#     return noise_level

# def apply_noise_reduction(image):
#     noise_level = calculate_noise_level(image)

#     if noise_level < 10:
#         # Apply low noise reduction function
#         reduced_noise_image = image
#     elif noise_level < 50:
#         # Apply regular noise reduction function
#         reduced_noise_image = reduce_noise_median(image)
#     else:
#         reduced_noise_image = reduce_noise_median(image)
#         reduced_noise_image = reduce_noise_morph(image)
    
#     return reduced_noise_image
    

# # Main code to process a single image file
# def process_single_image(image_path):
#     image = cv2.imread(image_path)
#     reduced_noise_image = apply_noise_reduction(image)
#     output_image, grey_img, bin_img = find_shapes(reduced_noise_image)
#     show_regular(reduced_noise_image, grey_img, bin_img, output_image, title="Shape Detection")

# # Example usage
# image_path = "images\isolated.png"  # Replace with your image path
# process_single_image(image_path)


# import numpy as np
# from matplotlib import pyplot as plt
# import cv2
# from functions import reduce_noise_median, reduce_noise_morph

# def detect_shape_and_approx(contour):
#     """Function to detect the shape and get the approximated contour."""
#     shape = "unidentified"
#     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

#     # Calculate aspect ratio and bounding box to help with rounded rectangle detection
#     x, y, w, h = cv2.boundingRect(approx)
#     ratio = float(w) / h

#     if len(approx) == 3:
#         shape = "triangle"
#     elif len(approx) == 4:
#         if ratio >= 0.95 and ratio <= 1.05:
#             shape = "square"
#         else:
#             shape = "rectangle"
#         # Check if the rectangle might be rounded
#         if cv2.contourArea(contour) > 0.5 * w * h:
#             shape = "rounded_rectangle"
#     elif len(approx) == 5:
#         shape = "rectangle"
#     elif len(approx) == 6:
#         shape = "hexagon"
#     elif len(approx) > 6 and len(approx) <= 10:
#         shape = "star"
#     else:
#         shape = "ellipse"
    
#     return shape, approx

# def replace_with_shape(approx, shape):
#     """Replace detected shape with its ideal form."""
#     if shape == "triangle":
#         ret, triangle = cv2.minEnclosingTriangle(approx)
#         return triangle.reshape(-1, 2).astype(int)
    
#     elif shape == "square" or shape == "rectangle":
#         rect = cv2.minAreaRect(approx)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         return box
#     elif shape == "rounded_rectangle":
#         # Approximating rounded rectangles with a box
#         rect = cv2.minAreaRect(approx)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         return box
#     elif shape == "pentagon" or shape == "hexagon":
#         center, radius = cv2.minEnclosingCircle(approx)
#         num_sides = 5 if shape == "pentagon" else 6
#         poly = []
#         for angle in np.linspace(0, 2 * np.pi, num=num_sides, endpoint=False):
#             poly.append([
#                 int(center[0] + radius * np.cos(angle)),
#                 int(center[1] + radius * np.sin(angle))
#             ])
#         return np.array(poly, dtype=int)
#     elif shape == "star":
#         # Improved star shape generation
#         center, radius = cv2.minEnclosingCircle(approx)
#         outer_points = []
#         inner_points = []
#         for i in range(5):
#             outer_angle = i * 2 * np.pi / 5 - np.pi / 2
#             inner_angle = outer_angle + np.pi / 5
            
#             outer_x = int(center[0] + radius * np.cos(outer_angle))
#             outer_y = int(center[1] + radius * np.sin(outer_angle))
#             outer_points.append([outer_x, outer_y])
            
#             inner_x = int(center[0] + 0.4 * radius * np.cos(inner_angle))
#             inner_y = int(center[1] + 0.4 * radius * np.sin(inner_angle))
#             inner_points.append([inner_x, inner_y])
        
#         star_points = []
#         for i in range(5):
#             star_points.append(outer_points[i])
#             star_points.append(inner_points[i])
        
#         return np.array(star_points, dtype=int)
#     elif shape == "ellipse":
#         ellipse = cv2.fitEllipse(approx)
#         ellipse_points = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), 
#                                           (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), 
#                                           int(ellipse[2]), 0, 360, 5)
#         return np.array(ellipse_points, dtype=int)
#     else:
#         return approx.reshape(-1, 2).astype(int)

# def check_symmetry(shape, points):
#     """Check for different types of symmetry in the given shape."""
#     def is_symmetric(pts, axis, direction='vertical'):
#         """Check if the shape is symmetric along a given axis (vertical or horizontal)."""
#         pts = np.array(pts)
#         if direction == 'vertical':
#             mirrored_pts = [[2 * axis - x, y] for x, y in pts]
#         else:
#             mirrored_pts = [[x, 2 * axis - y] for x, y in pts]
#         mirrored_pts = np.array(mirrored_pts)
#         # Sorting points to compare
#         return np.allclose(np.sort(pts, axis=0), np.sort(mirrored_pts, axis=0))

#     def is_rotationally_symmetric(pts):
#         """Check if the shape is rotationally symmetric."""
#         num_points = len(pts)
#         if num_points < 3:
#             return False  # Not enough points for meaningful rotational symmetry

#         # Compute center
#         center_x = np.mean(pts[:, 0])
#         center_y = np.mean(pts[:, 1])
        
#         # Compute angles and sort
#         angles = [np.arctan2(y - center_y, x - center_x) for x, y in pts]
#         angles = np.sort(angles)
        
#         # Check rotational symmetry
#         expected_angle = 2 * np.pi / num_points
#         for i in range(num_points):
#             rotated_angles = np.roll(angles, -i)  # Rotate angles
#             if np.allclose(rotated_angles, angles):
#                 return True
#         return False

#     center_x = np.mean(points[:, 0])
#     center_y = np.mean(points[:, 1])
    
#     symmetries = {}
    
#     if shape in ["rectangle", "square", "rounded_rectangle"]:
#         symmetries["vertical"] = is_symmetric(points, center_x, 'vertical')
#         symmetries["horizontal"] = is_symmetric(points, center_y, 'horizontal')
    
#     if shape in ["triangle", "pentagon", "hexagon"]:
#         symmetries["rotational"] = is_rotationally_symmetric(points)
    
#     if shape == "ellipse":
#         symmetries["vertical"] = is_symmetric(points, center_x, 'vertical')
#         symmetries["horizontal"] = is_symmetric(points, center_y, 'horizontal')
    
#     return symmetries



# def find_shapes(image_bgr):
#     image_output = image_bgr.copy()
#     image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)

#     # Apply edge detection
#     edges = cv2.Canny(image_gray, 50, 150)

#     # Optionally, you can still apply morphological operations to the edge image
#     kernel = np.ones((3, 3), np.uint8)
#     edges = cv2.dilate(edges, kernel, iterations=1)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Detect lines using Hough Line Transform
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=50, maxLineGap=10)
    
#     plt.figure(figsize=(10, 10))

#     # Draw lines on the output image
#     # if lines is not None:
#     #     for line in lines:
#     #         x1, y1, x2, y2 = line[0]
#     #         plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

#     for contour in contours:
#         if cv2.arcLength(contour, True) <= 100:
#             continue

#         # Detect the shape and approximate contour
#         shape, approx = detect_shape_and_approx(contour)

#         # Replace the contour with the ideal shape
#         ideal_shape = replace_with_shape(approx, shape)

#         # Flip y-coordinates
#         height, width = image_bgr.shape[:2]
#         ideal_shape_flipped = ideal_shape.copy()
#         ideal_shape_flipped[:, 1] = height - ideal_shape[:, 1]

#         # Check symmetry after replacing shape
#         symmetries = check_symmetry(shape, ideal_shape_flipped)
#         print(f"Shape: {shape}, Symmetries: {symmetries}")

#         # Plot the ideal shape with flipped y-coordinates
#         plt.plot(np.append(ideal_shape_flipped[:, 0], ideal_shape_flipped[0, 0]), 
#                  np.append(ideal_shape_flipped[:, 1], ideal_shape_flipped[0, 1]), 
#                  'g-', linewidth=2)

#     # Set the aspect ratio to be equal and remove axis
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.axis('off')

#     # Display the final replaced shapes
#     plt.show()

#     return image_output, image_gray, edges  # Return edges instead of binary image


# def show_regular(input_img, grey_img, edge_img, output_img, title="figure1"):
#     fig, _ = plt.subplots(figsize=(10, 10))
#     fig.canvas.setWindowTitle(title)

#     plt.subplot(221)
#     plt.axis('off')
#     plt.title("input")
#     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(222)
#     plt.axis('off')
#     plt.title("greyscale")
#     plt.imshow(grey_img, cmap='gray')

#     plt.subplot(223)
#     plt.axis('off')
#     plt.title("edges")
#     plt.imshow(edge_img, cmap='gray')

#     plt.subplot(224)
#     plt.axis('off')
#     plt.title("output")
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
#     plt.show()

#     return

# def calculate_noise_level(image):
#     noise_level = np.std(image)
#     return noise_level

# def apply_noise_reduction(image):
#     noise_level = calculate_noise_level(image)

#     if noise_level < 10:
#         # Apply low noise reduction function
#         reduced_noise_image = image
#     elif noise_level < 50:
#         # Apply regular noise reduction function
#         reduced_noise_image = reduce_noise_median(image)
#     else:
#         reduced_noise_image = reduce_noise_median(image)
#         reduced_noise_image = reduce_noise_morph(image)
    
#     return reduced_noise_image

# # Main code to process a single image file
# def process_single_image(image_path):
#     image = cv2.imread(image_path)
#     reduced_noise_image = apply_noise_reduction(image)
#     output_image, grey_img, edge_img = find_shapes(reduced_noise_image)
#     show_regular(reduced_noise_image, grey_img, edge_img, output_image, title="Shape Detection")

# # Example usage
# # image_path = "images/isolated.png"  # Replace with your image path


# import numpy as np
# import cv2
# from functions import reduce_noise_median, reduce_noise_morph
# import matplotlib
#  # Set the backend to non-interactive
# from matplotlib import pyplot as plt

# def detect_shape_and_approx(contour):
#     """Function to detect the shape and get the approximated contour."""
#     shape = "unidentified"
#     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

#     # Calculate aspect ratio and bounding box to help with rounded rectangle detection
#     x, y, w, h = cv2.boundingRect(approx)
#     ratio = float(w) / h

#     if len(approx) == 3:
#         shape = "triangle"
#     elif len(approx) == 4:
#         if ratio >= 0.95 and ratio <= 1.05:
#             shape = "square"
#         else:
#             shape = "rectangle"
#         # Check if the rectangle might be rounded
#         if cv2.contourArea(contour) > 0.5 * w * h:
#             shape = "rounded_rectangle"
#     elif len(approx) == 5:
#         shape = "rectangle"
#     elif len(approx) == 6:
#         shape = "hexagon"
#     elif len(approx) > 6 and len(approx) <= 10:
#         shape = "star"
#     else:
#         shape = "ellipse"
    
#     return shape, approx

# def replace_with_shape(approx, shape):
#     """Replace detected shape with its ideal form."""
#     if shape == "triangle":
#         ret, triangle = cv2.minEnclosingTriangle(approx)
#         return triangle.reshape(-1, 2).astype(int)
    
#     elif shape == "square" or shape == "rectangle":
#         rect = cv2.minAreaRect(approx)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         return box
#     elif shape == "rounded_rectangle":
#         # Approximating rounded rectangles with a box
#         rect = cv2.minAreaRect(approx)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         return box
#     elif shape == "pentagon" or shape == "hexagon":
#         center, radius = cv2.minEnclosingCircle(approx)
#         num_sides = 5 if shape == "pentagon" else 6
#         poly = []
#         for angle in np.linspace(0, 2 * np.pi, num=num_sides, endpoint=False):
#             poly.append([
#                 int(center[0] + radius * np.cos(angle)),
#                 int(center[1] + radius * np.sin(angle))
#             ])
#         return np.array(poly, dtype=int)
#     elif shape == "star":
#         # Improved star shape generation
#         center, radius = cv2.minEnclosingCircle(approx)
#         outer_points = []
#         inner_points = []
#         for i in range(5):
#             outer_angle = i * 2 * np.pi / 5 - np.pi / 2
#             inner_angle = outer_angle + np.pi / 5
            
#             outer_x = int(center[0] + radius * np.cos(outer_angle))
#             outer_y = int(center[1] + radius * np.sin(outer_angle))
#             outer_points.append([outer_x, outer_y])
            
#             inner_x = int(center[0] + 0.4 * radius * np.cos(inner_angle))
#             inner_y = int(center[1] + 0.4 * radius * np.sin(inner_angle))
#             inner_points.append([inner_x, inner_y])
        
#         star_points = []
#         for i in range(5):
#             star_points.append(outer_points[i])
#             star_points.append(inner_points[i])
        
#         return np.array(star_points, dtype=int)
#     elif shape == "ellipse":
#         ellipse = cv2.fitEllipse(approx)
#         ellipse_points = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), 
#                                           (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), 
#                                           int(ellipse[2]), 0, 360, 5)
#         return np.array(ellipse_points, dtype=int)
#     else:
#         return approx.reshape(-1, 2).astype(int)

# def check_symmetry(shape, points):
#     """Check for different types of symmetry in the given shape."""
#     def is_symmetric(pts, axis, direction='vertical'):
#         """Check if the shape is symmetric along a given axis (vertical or horizontal)."""
#         pts = np.array(pts)
#         if direction == 'vertical':
#             mirrored_pts = [[2 * axis - x, y] for x, y in pts]
#         else:
#             mirrored_pts = [[x, 2 * axis - y] for x, y in pts]
#         mirrored_pts = np.array(mirrored_pts)
#         # Sorting points to compare
#         return np.allclose(np.sort(pts, axis=0), np.sort(mirrored_pts, axis=0))

#     def is_rotationally_symmetric(pts):
#         """Check if the shape is rotationally symmetric."""
#         num_points = len(pts)
#         if num_points < 3:
#             return False  # Not enough points for meaningful rotational symmetry

#         # Compute center
#         center_x = np.mean(pts[:, 0])
#         center_y = np.mean(pts[:, 1])
        
#         # Compute angles and sort
#         angles = [np.arctan2(y - center_y, x - center_x) for x, y in pts]
#         angles = np.sort(angles)
        
#         # Check rotational symmetry
#         expected_angle = 2 * np.pi / num_points
#         for i in range(num_points):
#             rotated_angles = np.roll(angles, -i)  # Rotate angles
#             if np.allclose(rotated_angles, angles):
#                 return True
#         return False

#     center_x = np.mean(points[:, 0])
#     center_y = np.mean(points[:, 1])
    
#     symmetries = {}
    
#     if shape in ["rectangle", "square", "rounded_rectangle"]:
#         symmetries["vertical"] = is_symmetric(points, center_x, 'vertical')
#         symmetries["horizontal"] = is_symmetric(points, center_y, 'horizontal')
    
#     if shape in ["triangle", "pentagon", "hexagon"]:
#         symmetries["rotational"] = is_rotationally_symmetric(points)
    
#     if shape == "ellipse":
#         symmetries["vertical"] = is_symmetric(points, center_x, 'vertical')
#         symmetries["horizontal"] = is_symmetric(points, center_y, 'horizontal')
    
#     return symmetries

# def find_shapes(image_bgr):
#     image_output = image_bgr.copy()
#     image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)

#     edges = cv2.Canny(image_gray, 50, 150)

#     kernel = np.ones((3, 3), np.uint8)
#     edges = cv2.dilate(edges, kernel, iterations=1)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=50, maxLineGap=10)
    
#     plt.figure(figsize=(10, 10))

#     for contour in contours:
#         if cv2.arcLength(contour, True) <= 100:
#             continue

#         shape, approx = detect_shape_and_approx(contour)
#         ideal_shape = replace_with_shape(approx, shape)

#         height, width = image_bgr.shape[:2]
#         ideal_shape_flipped = ideal_shape.copy()
#         ideal_shape_flipped[:, 1] = height - ideal_shape[:, 1]

#         symmetries = check_symmetry(shape, ideal_shape_flipped)
#         print(f"Shape: {shape}, Symmetries: {symmetries}")

#         plt.plot(np.append(ideal_shape_flipped[:, 0], ideal_shape_flipped[0, 0]), 
#                  np.append(ideal_shape_flipped[:, 1], ideal_shape_flipped[0, 1]), 
#                  'g-', linewidth=2)

#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.axis('off')

#     # Save the plot instead of showing it
#     plt.savefig('shapes_detected.png')
#     plt.close()  # Close the figure to free up memory

#     return image_output, image_gray, edges

# def show_regular(input_img, grey_img, edge_img, output_img, title="figure1"):
#     fig, _ = plt.subplots(figsize=(10, 10))
#     fig.canvas.set_window_title(title)

#     plt.subplot(221)
#     plt.axis('off')
#     plt.title("input")
#     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(222)
#     plt.axis('off')
#     plt.title("greyscale")
#     plt.imshow(grey_img, cmap='gray')

#     plt.subplot(223)
#     plt.axis('off')
#     plt.title("edges")
#     plt.imshow(edge_img, cmap='gray')

#     plt.subplot(224)
#     plt.axis('off')
#     plt.title("output")
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    
#     # Save the plot instead of showing it
#     plt.savefig(f'{title}.png')
#     plt.close()  # Close the figure to free up memory

# # ... [All other functions remain the same] ...
# def calculate_noise_level(image):
#     noise_level = np.std(image)
#     return noise_level

# def apply_noise_reduction(image):
#     noise_level = calculate_noise_level(image)

#     if noise_level < 10:
#         # Apply low noise reduction function
#         reduced_noise_image = image
#     elif noise_level < 50:
#         # Apply regular noise reduction function
#         reduced_noise_image = reduce_noise_median(image)
#     else:
#         reduced_noise_image = reduce_noise_median(image)
#         reduced_noise_image = reduce_noise_morph(image)
    
#     return reduced_noise_image

# # Main code to process a single image file
# def process_single_image(image_path):
#     image = cv2.imread(image_path)
#     reduced_noise_image = apply_noise_reduction(image)
#     output_image, grey_img, edge_img = find_shapes(reduced_noise_image)
#     show_regular(reduced_noise_image, grey_img, edge_img, output_image, title="Shape_Detection")


import numpy as np
import cv2
from functions import reduce_noise_median, reduce_noise_morph
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
from matplotlib import pyplot as plt

def detect_shape_and_approx(contour):
    """Function to detect the shape and get the approximated contour."""
    shape = "unidentified"
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    # Calculate aspect ratio and bounding box to help with rounded rectangle detection
    x, y, w, h = cv2.boundingRect(approx)
    ratio = float(w) / h

    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        if ratio >= 0.95 and ratio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"
        # Check if the rectangle might be rounded
        if cv2.contourArea(contour) > 0.5 * w * h:
            shape = "rounded_rectangle"
    elif len(approx) == 5:
        shape = "rectangle"
    elif len(approx) == 6:
        shape = "hexagon"
    elif len(approx) > 6 and len(approx) <= 10:
        shape = "star"
    else:
        shape = "ellipse"
    
    return shape, approx

def replace_with_shape(approx, shape):
    """Replace detected shape with its ideal form."""
    if shape == "triangle":
        ret, triangle = cv2.minEnclosingTriangle(approx)
        return triangle.reshape(-1, 2).astype(int)
    
    elif shape == "square" or shape == "rectangle":
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box
    elif shape == "rounded_rectangle":
        # Approximating rounded rectangles with a box
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box
    elif shape == "pentagon" or shape == "hexagon":
        center, radius = cv2.minEnclosingCircle(approx)
        num_sides = 5 if shape == "pentagon" else 6
        poly = []
        for angle in np.linspace(0, 2 * np.pi, num=num_sides, endpoint=False):
            poly.append([
                int(center[0] + radius * np.cos(angle)),
                int(center[1] + radius * np.sin(angle))
            ])
        return np.array(poly, dtype=int)
    elif shape == "star":
        # Improved star shape generation
        center, radius = cv2.minEnclosingCircle(approx)
        outer_points = []
        inner_points = []
        for i in range(5):
            outer_angle = i * 2 * np.pi / 5 - np.pi / 2
            inner_angle = outer_angle + np.pi / 5
            
            outer_x = int(center[0] + radius * np.cos(outer_angle))
            outer_y = int(center[1] + radius * np.sin(outer_angle))
            outer_points.append([outer_x, outer_y])
            
            inner_x = int(center[0] + 0.4 * radius * np.cos(inner_angle))
            inner_y = int(center[1] + 0.4 * radius * np.sin(inner_angle))
            inner_points.append([inner_x, inner_y])
        
        star_points = []
        for i in range(5):
            star_points.append(outer_points[i])
            star_points.append(inner_points[i])
        
        return np.array(star_points, dtype=int)
    elif shape == "ellipse":
        ellipse = cv2.fitEllipse(approx)
        ellipse_points = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), 
                                          (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), 
                                          int(ellipse[2]), 0, 360, 5)
        return np.array(ellipse_points, dtype=int)
    else:
        return approx.reshape(-1, 2).astype(int)

def check_symmetry(shape, points):
    """Check for different types of symmetry in the given shape."""
    def is_symmetric(pts, axis, direction='vertical'):
        """Check if the shape is symmetric along a given axis (vertical or horizontal)."""
        pts = np.array(pts)
        if direction == 'vertical':
            mirrored_pts = [[2 * axis - x, y] for x, y in pts]
        else:
            mirrored_pts = [[x, 2 * axis - y] for x, y in pts]
        mirrored_pts = np.array(mirrored_pts)
        # Sorting points to compare
        return np.allclose(np.sort(pts, axis=0), np.sort(mirrored_pts, axis=0))

    def is_rotationally_symmetric(pts):
        """Check if the shape is rotationally symmetric."""
        num_points = len(pts)
        if num_points < 3:
            return False  # Not enough points for meaningful rotational symmetry

        # Compute center
        center_x = np.mean(pts[:, 0])
        center_y = np.mean(pts[:, 1])
        
        # Compute angles and sort
        angles = [np.arctan2(y - center_y, x - center_x) for x, y in pts]
        angles = np.sort(angles)
        
        # Check rotational symmetry
        expected_angle = 2 * np.pi / num_points
        for i in range(num_points):
            rotated_angles = np.roll(angles, -i)  # Rotate angles
            if np.allclose(rotated_angles, angles):
                return True
        return False

    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    
    symmetries = {}
    
    if shape in ["rectangle", "square", "rounded_rectangle"]:
        symmetries["vertical"] = is_symmetric(points, center_x, 'vertical')
        symmetries["horizontal"] = is_symmetric(points, center_y, 'horizontal')
    
    if shape in ["triangle", "pentagon", "hexagon"]:
        symmetries["rotational"] = is_rotationally_symmetric(points)
    
    if shape == "ellipse":
        symmetries["vertical"] = is_symmetric(points, center_x, 'vertical')
        symmetries["horizontal"] = is_symmetric(points, center_y, 'horizontal')
    
    return symmetries

def find_shapes(image_bgr):
    image_output = image_bgr.copy()
    image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image_gray, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=50, maxLineGap=10)
    
    plt.figure(figsize=(10, 10))

    for contour in contours:
        if cv2.arcLength(contour, True) <= 100:
            continue

        shape, approx = detect_shape_and_approx(contour)
        ideal_shape = replace_with_shape(approx, shape)

        height, width = image_bgr.shape[:2]
        ideal_shape_flipped = ideal_shape.copy()
        ideal_shape_flipped[:, 1] = height - ideal_shape[:, 1]

        symmetries = check_symmetry(shape, ideal_shape_flipped)
        print(f"Shape: {shape}, Symmetries: {symmetries}")

        plt.plot(np.append(ideal_shape_flipped[:, 0], ideal_shape_flipped[0, 0]), 
                 np.append(ideal_shape_flipped[:, 1], ideal_shape_flipped[0, 1]), 
                 'g-', linewidth=2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    # Save the plot instead of showing it
    plt.savefig('static/shapes_detected.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free up memory

    return image_output, image_gray, edges

def show_regular(input_img, grey_img, edge_img, output_img, title="figure1"):
    fig, _ = plt.subplots(figsize=(10, 10))

    plt.subplot(221)
    plt.axis('off')
    plt.title("input")
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

    plt.subplot(222)
    plt.axis('off')
    plt.title("greyscale")
    plt.imshow(grey_img, cmap='gray')

    plt.subplot(223)
    plt.axis('off')
    plt.title("edges")
    plt.imshow(edge_img, cmap='gray')

    plt.subplot(224)
    plt.axis('off')
    plt.title("output")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    
    # Save the plot instead of showing it
    plt.savefig(f'static/{title}.png', bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

def calculate_noise_level(image):
    noise_level = np.std(image)
    return noise_level

def apply_noise_reduction(image):
    noise_level = calculate_noise_level(image)

    if noise_level < 10:
        # Apply low noise reduction function
        reduced_noise_image = image
    elif noise_level < 50:
        # Apply regular noise reduction function
        reduced_noise_image = reduce_noise_median(image)
    else:
        reduced_noise_image = reduce_noise_median(image)
        reduced_noise_image = reduce_noise_morph(image)
    
    return reduced_noise_image

# Main code to process a single image file
def process_single_image(image_path):
    image = cv2.imread(image_path)
    reduced_noise_image = apply_noise_reduction(image)
    output_image, grey_img, edge_img = find_shapes(reduced_noise_image)
    show_regular(reduced_noise_image, grey_img, edge_img, output_image, title="Shape_Detection")
