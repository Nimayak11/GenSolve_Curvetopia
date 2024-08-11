from final_isolated import process_single_image
from func_copy_fit import process_single_image_frag
from occlusion import occlusion
from test import test

reply = input("Enter choice: ")
image_path = input("Enter path: ")
if (reply == "Isolated"):
    process_single_image(image_path)
elif (reply == "Fragmented"):
    process_single_image_frag(image_path)
elif (reply == "Occluded Shape Contours"):
    occlusion(image_path)
elif (reply == "Occluded Ellipse Completion"):
    test(image_path)
else:
    print("Invalid input")