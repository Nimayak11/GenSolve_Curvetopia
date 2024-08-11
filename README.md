### Here's the website link where you can test the algorithms used:

https://gensolve-curvetopia-2.onrender.com/


### Methodology:
Used for Shape Identification, Regularization & Symmetry hunt:<br/> The code integrates several methodologies for image processing and shape detection. Initially, it applies noise reduction techniques to the image, including median filtering and morphological operations, based on the calculated noise level. The preprocessed image is then subjected to edge detection using the Canny method, followed by dilation to enhance the edges. Contours are detected and approximated to simpler polygons using `cv2.approxPolyDP`, with shape classification performed based on the number of vertices and aspect ratios. The code identifies various shapes such as triangles, squares, rectangles, pentagons, hexagons, stars, and ellipses, replacing detected contours with idealized shapes. Symmetry is analyzed, including vertical, horizontal, and rotational symmetry, depending on the shape. The results are visualized using `matplotlib`, with shapes plotted and saved to files. The overall processing flow includes reading the image, applying noise reduction, detecting and classifying shapes, and generating visual outputs for further analysis.<br />
Note : Symmetry detection has only been introduced in the final_isolated python file. Although meant solely for images with isolated shapes, it can be used to find symmetry for the fragmented shapes as well. So you can choose the 'Isolated' option even on images with multiple overlapped shapes. The only difference is that the accuracy of shape detection and regularization using this option will be low for an input image with fragmented or overlapped shapes.<br />

For Occlusion : <br/>
The occlusion function processes an image to detect red regions by applying color filtering and contour detection, highlighting the largest contour and extracting the foreground region. The DTECMA function performs connected component analysis on a binary image, identifying and classifying ellipses based on various parameters like distance threshold, ratio, and angle. It utilizes additional helper functions for ellipse detection, distance computation, and integer programming to optimize ellipse fitting. The yy and xx functions generate coordinate grids for image processing, while elipsBound calculates bounding box coordinates for ellipses. concaveDetect identifies concave points in polygons, and findEllipse detects ellipses in an image using geometric transformations and distance metrics. The overlay function merges ellipse data with contour images, and computeDist calculates distances between contours and ellipses. Finally, intergerProgamming uses integer programming to optimize ellipse selection, and plotEllipse visualizes the ellipses by overlaying them on the input image. This is based on the paper: [Recognition of overlapping elliptical objects in a binary image](https://link.springer.com/article/10.1007/s10044-020-00951-z).

<br/>

### How to use the code??:

To run locally and to view the symmetries detected, you'll need to download the files of the repository and then run the Adobe_main python file.<br/>

Otherwise you can directly use the website.<br/><br/>
For demo, input from the images folder yield the optimal output but you can also run the codes on any other file of your choice.
