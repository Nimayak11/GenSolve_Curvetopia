### Here's the website link where you can test the algorithms used:

https://gensolve-curvetopia-2.onrender.com/


### Methodology:
Used for Shape Identification, Regularization & Symmetry hunt: The code integrates several methodologies for image processing and shape detection. Initially, it applies noise reduction techniques to the image, including median filtering and morphological operations, based on the calculated noise level. The preprocessed image is then subjected to edge detection using the Canny method, followed by dilation to enhance the edges. Contours are detected and approximated to simpler polygons using `cv2.approxPolyDP`, with shape classification performed based on the number of vertices and aspect ratios. The code identifies various shapes such as triangles, squares, rectangles, pentagons, hexagons, stars, and ellipses, replacing detected contours with idealized shapes. Symmetry is analyzed, including vertical, horizontal, and rotational symmetry, depending on the shape. The results are visualized using `matplotlib`, with shapes plotted and saved to files. The overall processing flow includes reading the image, applying noise reduction, detecting and classifying shapes, and generating visual outputs for further analysis.\
Note : Symmetry detection has only been introduced in the final_isolated python file. Although meant solely for images with isolated shapes, it can be used to find symmetry for the fragmented shapes as well. So you can choose the 'Isolated' option even on images with multiple overlapped shapes. The only difference is that the accuracy of shape detection and regularization using this option will be low for an input image with fragmented or overlapped shapes.\

For Occlusion : 
