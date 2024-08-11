
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
        
        return np.allclose(np.sort(pts, axis=0), np.sort(mirrored_pts, axis=0))

    def is_rotationally_symmetric(pts):
        """Check if the shape is rotationally symmetric."""
        num_points = len(pts)
        if num_points < 3:
            return False  # Not enough points for meaningful rotational symmetry

        
        center_x = np.mean(pts[:, 0])
        center_y = np.mean(pts[:, 1])
        
        
        angles = [np.arctan2(y - center_y, x - center_x) for x, y in pts]
        angles = np.sort(angles)
        
        
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
