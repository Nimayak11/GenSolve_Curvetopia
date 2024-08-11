from flask import Flask, request, send_file
from final_isolated import process_single_image
from func_copy_fit import process_single_image_frag
from occlusion import occlusion
from test import test
import os

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return '''
#         <h1>Choose an option</h1>
#         <form action="/process" method="post" enctype="multipart/form-data">
#             <label for="choice">Choose:</label>
#             <select id="choice" name="choice">
#                 <option value="Isolated">Isolated</option>
#                 <option value="Fragmented">Fragmented</option>
#                 <option value="Occluded Shape Contours">Occluded Shape Contours</option>
#                 <option value="Occluded Ellipse Completion">Occluded Ellipse Completion</option>
#             </select><br><br>
#             <label for="file">Select Image:</label>
#             <input type="file" id="file" name="file"><br><br>
#             <input type="submit" value="Submit">
#         </form>
#     '''

# @app.route('/process', methods=['POST'])
# def process():
#     choice = request.form['choice']
#     file = request.files['file']
#     if file:
#         image_path = f"./uploads/{file.filename}"
#         file.save(image_path)

#         # Process the image based on user choice
#         if choice == "Isolated":
#             result_image, _, _ = process_single_image(image_path)
#             return send_file(result_image, mimetype='image/png')
#         elif choice == "Fragmented":
#             result_image, _, _ = process_single_image_frag(image_path)
#             return send_file(result_image, mimetype='image/png')
#         elif choice == "Occluded Shape Contours":
#             result_image, _, _ = occlusion(image_path)
#             return send_file(result_image, mimetype='image/png')
#         elif choice == "Occluded Ellipse Completion":
#             result_image, _, _ = test(image_path)
#             return send_file(result_image, mimetype='image/png')
#         else:
#             return "Invalid choice"

#         # Return the processed image file
#         return send_file(result_image, mimetype='image/png')

#     return "No file uploaded"

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <form action="/process" method="post" enctype="multipart/form-data">
        <label for="image">Upload image:</label>
        <input type="file" name="image" id="image" required>
        <label for="choice">Choose an option:</label>
        <select name="choice" id="choice">
            <option value="Isolated">Isolated</option>
            <option value="Fragmented">Fragmented</option>
            <option value="Occluded Shape Contours">Occluded Shape Contours</option>
            <option value="Occluded Ellipse Completion">Occluded Ellipse Completion</option>
        </select>
        <input type="submit" value="Submit">
    </form>
    '''

@app.route('/process', methods=['POST'])
def process():
    choice = request.form['choice']
    image = request.files['image']
    image_path = os.path.join('static', 'uploaded_image.png')
    image.save(image_path)

    
    if choice == "Isolated":
        process_single_image(image_path)
        result_image_path = 'static\shapes_detected.png'
    elif choice == "Fragmented":
        process_single_image_frag(image_path)
        result_image_path = 'static\Shape_Detection.png'
    elif choice == "Occluded Shape Contours":
        occlusion(image_path)
        result_image_path = 'static\occlusion_result.png'
    elif choice == "Occluded Ellipse Completion":
        test(image_path)
        result_image_path = 'static\output_completion.png'
    else:
        return "Invalid input", 400

   
    
    return send_file(result_image_path)

if __name__ == '__main__':
    app.run(debug=True)
