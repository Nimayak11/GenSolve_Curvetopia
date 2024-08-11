# import numpy as np
# import cv2
# from scipy.interpolate import splprep, splev
# from matplotlib import pyplot as plt
# from functions import reduce_noise_median, reduce_noise_morph

# def smooth_contour(contour, epsilon=0.01):
#     perimeter = cv2.arcLength(contour, True)
#     return cv2.approxPolyDP(contour, epsilon * perimeter, True)

# def fit_ellipse(contour):
#     if len(contour) >= 5:
#         return cv2.fitEllipse(contour)
#     return None

# def fit_spline(points, num_points=100):
#     if len(points) < 3:
#         # If not enough points, return the original points
#         return points

#     x, y = points.T
#     t = np.linspace(0, 1, len(x))

#     # Fit spline
#     tck, u = splprep([x, y], s=0, k=2)

#     # Generate new points
#     new_points = splev(np.linspace(0, 1, num_points), tck)
#     return np.column_stack(new_points).astype(np.int32)

# def find_shapes(image_bgr):
#     image_output = image_bgr.copy()
#     width = image_bgr.shape[1]
#     image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
#     _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     image_edges = cv2.Canny(image_binary, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(image_edges, 1, np.pi/180, 200, minLineLength=30, maxLineGap=10)

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             points = np.array([[x1, y1], [x2, y2]])
#             smooth_line = fit_spline(points)
#             cv2.polylines(image_output, [smooth_line], False, (0, 0, 255), 2)
#             cv2.putText(image_output, "Line", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

#     for contour in contours:
#         if cv2.arcLength(contour, True) <= 100:
#             continue
        
#         smooth_contour_pts = smooth_contour(contour)
#         x = smooth_contour_pts.ravel()[0] - 45
#         y = smooth_contour_pts.ravel()[1] + 80

#         if len(smooth_contour_pts) == 3:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (255, 0, 0), -1)
#             cv2.putText(image_output, "Triangle", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#         elif len(smooth_contour_pts) == 4:
#             x1, y1, w, h = cv2.boundingRect(smooth_contour_pts)
#             if w == width:
#                 continue
#             ratio = float(w) / h
#             if 0.95 <= ratio <= 1.05:
#                 cv2.drawContours(image_output, [smooth_contour_pts], 0, (205, 205, 0), -1)
#                 cv2.putText(image_output, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#             else:
#                 cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 0, 255), -1)
#                 cv2.putText(image_output, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#         elif len(smooth_contour_pts) == 5:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 140, 255), -1)
#             cv2.putText(image_output, "Pentagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#         elif len(smooth_contour_pts) == 6:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (219, 112, 147), -1)
#             cv2.putText(image_output, "Hexagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#         elif len(smooth_contour_pts) == 10:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 255, 255), -1)
#             cv2.putText(image_output, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#         else:
#             ellipse = fit_ellipse(smooth_contour_pts)
#             if ellipse is not None:
#                 cv2.ellipse(image_output, ellipse, (0, 255, 0), 2)
#                 cv2.putText(image_output, "Ellipse", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#             else:
#                 cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 255, 0), 2)
#                 cv2.putText(image_output, "Unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

#     return image_output, image_gray, image_binary

# def show_regular(input_img, grey_img, bin_img, output_img, title="figure1"):
#     fig, _ = plt.subplots(figsize=(10, 10))
#     fig.canvas.manager.set_window_title(title)

#     plt.subplot(221)
#     plt.axis('off')
#     plt.title("Input")
#     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(222)
#     plt.axis('off')
#     plt.title("Greyscale")
#     plt.imshow(cv2.cvtColor(grey_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(223)
#     plt.axis('off')
#     plt.title("Binary")
#     plt.imshow(cv2.cvtColor(bin_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(224)
#     plt.axis('off')
#     plt.title("Output")
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
#     plt.show()

#     return

# def calculate_noise_level(image):
#     noise_level = np.std(image)
#     return noise_level

# def apply_noise_reduction(image):
#     noise_level = calculate_noise_level(image)

#     if noise_level < 10:
#         reduced_noise_image = image
#     elif noise_level < 50:
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
# image_path = "images/frag2.png"  # Replace with your image path
# process_single_image(image_path)


# def find_shapes(image_bgr):
#     image_output = image_bgr.copy()
#     width = image_bgr.shape[1]
#     image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
#     _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     image_edges = cv2.Canny(image_binary, 50, 150, apertureSize = 3)
#     lines = cv2.HoughLinesP(image_edges, 1, np.pi/180, 200, minLineLength=30, maxLineGap=10)

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             points = np.array([[x1, y1], [x2, y2]])
#             smooth_line = fit_spline(points)
#             cv2.polylines(image_output, [smooth_line], False, (0, 0, 255), 2)
#             cv2.putText(image_output, "Line", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

#     for contour in contours:
#         if cv2.arcLength(contour, True) <= 100:
#             continue
        
#         smooth_contour = smooth_contour(contour)
#         x = smooth_contour.ravel()[0] - 45
#         y = smooth_contour.ravel()[1] + 80

#         if len(smooth_contour) == 3:
#             cv2.drawContours(image_output, [smooth_contour], 0, (255, 0, 0), -1)
#             cv2.putText(image_output, "Triangle", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#         elif len(smooth_contour) == 4:
#             x1, y1, w, h = cv2.boundingRect(smooth_contour)
#             if w == width:
#                 continue
#             ratio = float(w)/h
#             if 0.95 <= ratio <= 1.05:
#                 cv2.drawContours(image_output, [smooth_contour], 0, (205, 205, 0), -1)
#                 cv2.putText(image_output, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#             else:
#                 cv2.drawContours(image_output, [smooth_contour], 0, (0, 0, 255), -1)
#                 cv2.putText(image_output, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#         elif len(smooth_contour) == 5:
#             cv2.drawContours(image_output, [smooth_contour], 0, (0, 140, 255), -1)
#             cv2.putText(image_output, "Pentagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#         elif len(smooth_contour) == 6:
#             cv2.drawContours(image_output, [smooth_contour], 0, (219, 112, 147), -1)
#             cv2.putText(image_output, "Hexagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#         elif len(smooth_contour) == 10:
#             cv2.drawContours(image_output, [smooth_contour], 0, (0, 255, 255), -1)
#             cv2.putText(image_output, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#         else:
#             ellipse = fit_ellipse(smooth_contour)
#             if ellipse is not None:
#                 cv2.ellipse(image_output, ellipse, (0, 255, 0), 2)
#                 cv2.putText(image_output, "Ellipse", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#             else:
#                 cv2.drawContours(image_output, [smooth_contour], 0, (0, 255, 0), 2)
#                 cv2.putText(image_output, "Unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

#     return image_output, image_gray, image_binary

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
# image_path = "images/frag0.png"  # Replace with your image path
# process_single_image(image_path)

# import numpy as np
# import cv2
# from scipy.interpolate import splprep, splev
# from matplotlib import pyplot as plt
# from functions import reduce_noise_median, reduce_noise_morph

# def smooth_contour(contour, epsilon=0.01):
#     perimeter = cv2.arcLength(contour, True)
#     return cv2.approxPolyDP(contour, epsilon * perimeter, True)

# def fit_ellipse(contour):
#     if len(contour) >= 5:
#         return cv2.fitEllipse(contour)
#     return None

# def fit_spline(points, num_points=100):
#     if len(points) < 3:
#         # If not enough points, return the original points
#         return points

#     x, y = points.T
#     t = np.linspace(0, 1, len(x))

#     # Fit spline
#     tck, u = splprep([x, y], s=0, k=2)

#     # Generate new points
#     new_points = splev(np.linspace(0, 1, num_points), tck)
#     return np.column_stack(new_points).astype(np.int32)

# def find_shapes(image_bgr):
#     image_output = image_bgr.copy()
#     width = image_bgr.shape[1]
#     image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
#     _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     image_edges = cv2.Canny(image_binary, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(image_edges, 1, np.pi/180, 200, minLineLength=30, maxLineGap=10)

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             points = np.array([[x1, y1], [x2, y2]])
#             smooth_line = fit_spline(points)
#             cv2.polylines(image_output, [smooth_line], False, (0, 0, 255), 2)
#             cv2.putText(image_output, "Line", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#         print("Straight lines")

#     for contour in contours:
#         if cv2.arcLength(contour, True) <= 100:
#             continue
        
#         smooth_contour_pts = smooth_contour(contour)
#         x = smooth_contour_pts.ravel()[0] - 45
#         y = smooth_contour_pts.ravel()[1] + 80

#         if len(smooth_contour_pts) == 3:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (255, 0, 0), -1)
#             cv2.putText(image_output, "Triangle", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#             print("Triangle")
#         elif len(smooth_contour_pts) == 4:
#             x1, y1, w, h = cv2.boundingRect(smooth_contour_pts)
#             if w == width:
#                 continue
#             ratio = float(w) / h
#             if 0.95 <= ratio <= 1.05:
#                 cv2.drawContours(image_output, [smooth_contour_pts], 0, (205, 205, 0), -1)
#                 cv2.putText(image_output, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#                 print("Square")
#             else:
#                 cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 0, 255), -1)
#                 cv2.putText(image_output, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#                 print("Rectangle")
#         elif len(smooth_contour_pts) == 5:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 140, 255), -1)
#             cv2.putText(image_output, "Pentagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#             print("Pentagon/Rectangle")
#         elif len(smooth_contour_pts) == 6:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (219, 112, 147), -1)
#             cv2.putText(image_output, "Hexagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#             print("Hexagon")
#         elif len(smooth_contour_pts) == 10:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 255, 255), -1)
#             cv2.putText(image_output, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#             print("Star")
#         else:
#             ellipse = fit_ellipse(smooth_contour_pts)
#             if ellipse is not None:
#                 cv2.ellipse(image_output, ellipse, (0, 255, 0), 2)
#                 cv2.putText(image_output, "Ellipse", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#                 print("Ellipse")
#             else:
#                 cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 255, 0), 2)
#                 cv2.putText(image_output, "Unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#                 print("Unknown")

#     return image_output, image_gray, image_binary

# def show_regular(input_img, output_img, title="figure1"):
#     fig, _ = plt.subplots(figsize=(10, 10))
#     fig.canvas.manager.set_window_title(title)

#     plt.subplot(121)
#     plt.axis('off')
#     plt.title("Input")
#     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

#     # plt.subplot(222)
#     # plt.axis('off')
#     # plt.title("Greyscale")
#     # plt.imshow(cv2.cvtColor(grey_img, cv2.COLOR_BGR2RGB))

#     # plt.subplot(223)
#     # plt.axis('off')
#     # plt.title("Binary")
#     # plt.imshow(cv2.cvtColor(bin_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(122)
#     plt.axis('off')
#     plt.title("Output")
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
#     plt.show()

#     return

# def calculate_noise_level(image):
#     noise_level = np.std(image)
#     return noise_level

# def apply_noise_reduction(image):
#     noise_level = calculate_noise_level(image)

#     if noise_level < 10:
#         reduced_noise_image = image
#     elif noise_level < 50:
#         reduced_noise_image = reduce_noise_median(image)
#     else:
#         reduced_noise_image = reduce_noise_median(image)
#         reduced_noise_image = reduce_noise_morph(image)
    
#     return reduced_noise_image
    

# # Main code to process a single image file
# def process_single_image_frag(image_path):
#     image = cv2.imread(image_path)
#     reduced_noise_image = apply_noise_reduction(image)
#     output_image, grey_img, bin_img = find_shapes(reduced_noise_image)
#     show_regular(reduced_noise_image, output_image, title="Shape Detection")

# # Example usage
# # Replace with your image path



# # import numpy as np
# # import cv2
# # from scipy.interpolate import splprep, splev
# # from matplotlib import pyplot as plt

# # def smooth_contour(contour, epsilon=0.01):
# #     perimeter = cv2.arcLength(contour, True)
# #     return cv2.approxPolyDP(contour, epsilon * perimeter, True)

# # def fit_spline(points, num_points=100):
# #     x, y = points.T
# #     t = np.linspace(0, 1, len(x))
    
# #     # Fit spline
# #     tck, _ = splprep([x, y], s=0, k=2)
    
# #     # Generate new points
# #     new_points = splev(np.linspace(0, 1, num_points), tck)
# #     return np.column_stack(new_points).astype(np.int32)

# # def find_shapes(image_bgr):
# #     image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# #     _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
# #     contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# #     output_image = np.zeros_like(image_bgr)

# #     for contour in contours:
# #         if cv2.arcLength(contour, True) <= 100:
# #             continue
        
# #         smoothed_contour = smooth_contour(contour)
# #         if len(smoothed_contour) < 5:
# #             smooth_line = fit_spline(smoothed_contour[:, 0, :])
# #             cv2.polylines(output_image, [smooth_line], False, (0, 0, 255), 2)
# #             cv2.putText(output_image, "Line", tuple(smoothed_contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
# #         else:
# #             ellipse = cv2.fitEllipse(smoothed_contour)
# #             cv2.ellipse(output_image, ellipse, (0, 255, 0), 2)
# #             cv2.putText(output_image, "Ellipse", tuple(smoothed_contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# #     return output_image, image_gray, image_binary

# # def show_images(grey_img, bin_img, output_img, title="Figure"):
# #     plt.figure(figsize=(10, 10))
    
# #     plt.subplot(221)
# #     plt.axis('off')
# #     plt.title("Greyscale")
# #     plt.imshow(cv2.cvtColor(grey_img, cv2.COLOR_GRAY2RGB))
    
# #     plt.subplot(222)
# #     plt.axis('off')
# #     plt.title("Binary")
# #     plt.imshow(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB))

# #     plt.subplot(223)
# #     plt.axis('off')
# #     plt.title("Output")
# #     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    
# #     plt.show()

# # def process_single_image(image_path):
# #     image = cv2.imread(image_path)
# #     output_image, grey_img, bin_img = find_shapes(image)
# #     show_images(grey_img, bin_img, output_image, title="Shape Detection")

# # # Example usage
# # image_path = "images/isolated.png"  # Replace with your image path
# # process_single_image(image_path)


# import numpy as np
# import cv2
# from scipy.interpolate import splprep, splev
# import matplotlib
# # matplotlib.use('Agg')  # Set the backend to non-interactive
# from matplotlib import pyplot as plt
# from functions import reduce_noise_median, reduce_noise_morph

# def smooth_contour(contour, epsilon=0.01):
#     perimeter = cv2.arcLength(contour, True)
#     return cv2.approxPolyDP(contour, epsilon * perimeter, True)

# def fit_ellipse(contour):
#     if len(contour) >= 5:
#         return cv2.fitEllipse(contour)
#     return None

# def fit_spline(points, num_points=100):
#     if len(points) < 3:
#         return points

#     x, y = points.T
#     t = np.linspace(0, 1, len(x))

#     tck, u = splprep([x, y], s=0, k=2)

#     new_points = splev(np.linspace(0, 1, num_points), tck)
#     return np.column_stack(new_points).astype(np.int32)

# def find_shapes(image_bgr):
#     image_output = image_bgr.copy()
#     width = image_bgr.shape[1]
#     image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
#     _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     image_edges = cv2.Canny(image_binary, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(image_edges, 1, np.pi/180, 200, minLineLength=30, maxLineGap=10)

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             points = np.array([[x1, y1], [x2, y2]])
#             smooth_line = fit_spline(points)
#             cv2.polylines(image_output, [smooth_line], False, (0, 0, 255), 2)
#             cv2.putText(image_output, "Line", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#         print("Straight lines")

#     for contour in contours:
#         if cv2.arcLength(contour, True) <= 100:
#             continue
        
#         smooth_contour_pts = smooth_contour(contour)
#         x = smooth_contour_pts.ravel()[0] - 45
#         y = smooth_contour_pts.ravel()[1] + 80

#         if len(smooth_contour_pts) == 3:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (255, 0, 0), -1)
#             cv2.putText(image_output, "Triangle", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#             print("Triangle")
#         elif len(smooth_contour_pts) == 4:
#             x1, y1, w, h = cv2.boundingRect(smooth_contour_pts)
#             if w == width:
#                 continue
#             ratio = float(w) / h
#             if 0.95 <= ratio <= 1.05:
#                 cv2.drawContours(image_output, [smooth_contour_pts], 0, (205, 205, 0), -1)
#                 cv2.putText(image_output, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#                 print("Square")
#             else:
#                 cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 0, 255), -1)
#                 cv2.putText(image_output, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#                 print("Rectangle")
#         elif len(smooth_contour_pts) == 5:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 140, 255), -1)
#             cv2.putText(image_output, "Pentagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#             print("Pentagon/Rectangle")
#         elif len(smooth_contour_pts) == 6:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (219, 112, 147), -1)
#             cv2.putText(image_output, "Hexagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
#             print("Hexagon")
#         elif len(smooth_contour_pts) == 10:
#             cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 255, 255), -1)
#             cv2.putText(image_output, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#             print("Star")
#         else:
#             ellipse = fit_ellipse(smooth_contour_pts)
#             if ellipse is not None:
#                 cv2.ellipse(image_output, ellipse, (0, 255, 0), 2)
#                 cv2.putText(image_output, "Ellipse", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#                 print("Ellipse")
#             else:
#                 cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 255, 0), 2)
#                 cv2.putText(image_output, "Unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
#                 print("Unknown")

#     return image_output, image_gray, image_binary

# def show_regular(input_img, output_img, title="figure1"):
#     fig, _ = plt.subplots(figsize=(10, 10))

#     plt.subplot(121)
#     plt.axis('off')
#     plt.title("Input")
#     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

#     plt.subplot(122)
#     plt.axis('off')
#     plt.title("Output")
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    
#     # Save the plot instead of showing it
#     plt.savefig(f'{title}.png')
#     plt.close()  # Close the figure to free up memory

# def calculate_noise_level(image):
#     noise_level = np.std(image)
#     return noise_level

# def apply_noise_reduction(image):
#     noise_level = calculate_noise_level(image)

#     if noise_level < 10:
#         reduced_noise_image = image
#     elif noise_level < 50:
#         reduced_noise_image = reduce_noise_median(image)
#     else:
#         reduced_noise_image = reduce_noise_median(image)
#         reduced_noise_image = reduce_noise_morph(image)
    
#     return reduced_noise_image

# def process_single_image_frag(image_path):
#     image = cv2.imread(image_path)
#     reduced_noise_image = apply_noise_reduction(image)
#     output_image, grey_img, bin_img = find_shapes(reduced_noise_image)
#     show_regular(reduced_noise_image, output_image, title="Shape_Detection")

# # Example usage
# # Replace with your image path


import numpy as np
import cv2
from scipy.interpolate import splprep, splev
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
from matplotlib import pyplot as plt
from functions import reduce_noise_median, reduce_noise_morph

def smooth_contour(contour, epsilon=0.01):
    perimeter = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon * perimeter, True)

def fit_ellipse(contour):
    if len(contour) >= 5:
        return cv2.fitEllipse(contour)
    return None

def fit_spline(points, num_points=100):
    if len(points) < 3:
        return points

    x, y = points.T
    t = np.linspace(0, 1, len(x))

    tck, u = splprep([x, y], s=0, k=2)

    new_points = splev(np.linspace(0, 1, num_points), tck)
    return np.column_stack(new_points).astype(np.int32)

def find_shapes(image_bgr):
    image_output = image_bgr.copy()
    width = image_bgr.shape[1]
    image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    image_edges = cv2.Canny(image_binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(image_edges, 1, np.pi/180, 200, minLineLength=30, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points = np.array([[x1, y1], [x2, y2]])
            smooth_line = fit_spline(points)
            cv2.polylines(image_output, [smooth_line], False, (0, 0, 255), 2)
            cv2.putText(image_output, "Line", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        print("Straight lines")

    for contour in contours:
        if cv2.arcLength(contour, True) <= 100:
            continue
        
        smooth_contour_pts = smooth_contour(contour)
        x = smooth_contour_pts.ravel()[0] - 45
        y = smooth_contour_pts.ravel()[1] + 80

        if len(smooth_contour_pts) == 3:
            cv2.drawContours(image_output, [smooth_contour_pts], 0, (255, 0, 0), -1)
            cv2.putText(image_output, "Triangle", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
            print("Triangle")
        elif len(smooth_contour_pts) == 4:
            x1, y1, w, h = cv2.boundingRect(smooth_contour_pts)
            if w == width:
                continue
            ratio = float(w) / h
            if 0.95 <= ratio <= 1.05:
                cv2.drawContours(image_output, [smooth_contour_pts], 0, (205, 205, 0), -1)
                cv2.putText(image_output, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
                print("Square")
            else:
                cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 0, 255), -1)
                cv2.putText(image_output, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
                print("Rectangle")
        elif len(smooth_contour_pts) == 5:
            cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 140, 255), -1)
            cv2.putText(image_output, "Pentagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
            print("Pentagon/Rectangle")
        elif len(smooth_contour_pts) == 6:
            cv2.drawContours(image_output, [smooth_contour_pts], 0, (219, 112, 147), -1)
            cv2.putText(image_output, "Hexagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
            print("Hexagon")
        elif len(smooth_contour_pts) == 10:
            cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 255, 255), -1)
            cv2.putText(image_output, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
            print("Star")
        else:
            ellipse = fit_ellipse(smooth_contour_pts)
            if ellipse is not None:
                cv2.ellipse(image_output, ellipse, (0, 255, 0), 2)
                cv2.putText(image_output, "Ellipse", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
                print("Ellipse")
            else:
                cv2.drawContours(image_output, [smooth_contour_pts], 0, (0, 255, 0), 2)
                cv2.putText(image_output, "Unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
                print("Unknown")

    return image_output, image_gray, image_binary

def show_regular(input_img, output_img, title="figure1"):
    fig, _ = plt.subplots(figsize=(10, 10))

    plt.subplot(121)
    plt.axis('off')
    plt.title("Input")
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

    plt.subplot(122)
    plt.axis('off')
    plt.title("Output")
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
        reduced_noise_image = image
    elif noise_level < 50:
        reduced_noise_image = reduce_noise_median(image)
    else:
        reduced_noise_image = reduce_noise_median(image)
        reduced_noise_image = reduce_noise_morph(image)
    
    return reduced_noise_image

def process_single_image_frag(image_path):
    image = cv2.imread(image_path)
    reduced_noise_image = apply_noise_reduction(image)
    output_image, grey_img, bin_img = find_shapes(reduced_noise_image)
    show_regular(reduced_noise_image, output_image, title="Shape_Detection")
