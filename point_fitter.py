import warnings
import os
import cv2
import mediapipe as mp

import calculate_distances

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic


def fit_points():  # fit a mesh to the face
    # user inputs
    # The programme is going to assess whether the given file is real, by checking the path it is given
    while True:
        user_input = input("Enter the path of your image: ")
        if not os.path.exists(user_input):
            print("File not found at " + str(user_input))  # if no file exists, return the instruction that the file was not found
            continue
        else:
            path = user_input
            break
    while True:
        # For QC, the user can have a mesh drawn over the image
        # If the user does want a mesh, then it will be drawn and saved as a separate file in the same directory as the original
        user_input = input("Would you like a drawn mesh? Y/N: ")
        user_input = user_input.lower()
        if user_input != "y" and user_input != "n":
            print("Please enter Y or N")
            continue
        else:
            if user_input == "y":
                draw_mesh = True
            else:
                draw_mesh = False
            break
    # end of user inputs
    return fit_mesh(draw_mesh, path)  # Taking the single image input as a path, the image is passed for mesh fitting


def fit_multiple_images():
    # user inputs
    # The programme is going to assess whether the given folder is real, by checking the path it is given
    while True:
        user_input = input("Enter the path of your folder: ")
        if not os.path.exists(user_input):
            print("Folder not found at " + str(user_input))
            continue
        else:
            path = user_input
            break
    while True:
        # For QC, the user can have a mesh drawn over the image
        # If the user does want a mesh, then it will be drawn and saved as a separate file in the same directory as the original
        user_input = input("Would you like a drawn mesh? Y/N: ")
        user_input = user_input.lower()
        if user_input != "y" and user_input != "n":
            print("Please enter Y or N")
            continue
        else:
            if user_input == "y":
                draw_mesh = True
            else:
                draw_mesh = False
            break
    # end of user inputs
    i = 0  # Iteration for every file in the folder, this should end up being equal to the number of files in the folder
    j = 0  # Iteration for every valid image in the folder
    measurements_list = []
    # path generation
    # As the user has selected to do a bulk measuring, each file that can be found in the folder will be assessed
    # If the file is an image, and that image contains a face, it will be processed, otherwise it will be skipped
    for filename in os.listdir(path):
        print(filename)
        i += 1
        image_path = path + '/' + filename
        landmarks = fit_mesh(draw_mesh, image_path)  # This works the same as the single image processing
        if bool(landmarks):  # This checks if the image was able to be measured, if it was, then j increases
            # method to make measurements
            returned_dict = calculate_distances.calculate(landmarks)
            # The measurements are prepared as a dictionary of all measurements associated with that face
            # As multiple files are being processed, measurements_list is a list of all of these dictionaries
            returned_dict["ImageName"] = filename  # This allows for human readability for the csv output, by having the original filename be included in the dictionary of measurements
            measurements_list.extend([returned_dict])
            j += 1

    print(str(j) + ' images found and processed out of a total of ' + str(i) + ' files found')  # This shows the user how many files were successfully processed, for immediate QA
    return measurements_list


def fit_mesh(draw_mesh, path):
    # set the parameters
    # These parameters establish what sort of image the programme is expecting
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,  # The image is a still, not a video or an animation
            max_num_faces=1,  # There should only be one face present in the image
            refine_landmarks=True,  # The landmarks around the eyes are further refined for greater accuracy
            min_detection_confidence=0.5) as face_mesh:  # A heuristic detection confidence value that is found to work best for this task
        # open the image
        image = cv2.imread(path)
        # get the image dimensions
        h, w, _ = image.shape

        # apply the mesh
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # set a bool on whether the results are invalid (at this stage a valid result is just one that contains some data)
    face_found = bool(results.multi_face_landmarks)

    if face_found:  # if landmark data has landmarks then a face was found, move to calculate, and possibly draw points

        if draw_mesh:  # if the end user wants a digital representation of the points mapped to the face this will draw the mesh as a new file, otherwise this is skipped
            draw_with_points(image, results, path)

        landmarks = results.multi_face_landmarks[0].landmark  # create a list of each x, y, z tuple

        for landmark in landmarks:
            # multiply out the results of the co-ordinates by the width and height to un-normalize them
            landmark.x *= w
            landmark.y *= h

        return results.multi_face_landmarks[0].landmark  # return the results to main

    else:  # if no face is found, return warning
        user_warning('Failed to find a face at ' + path)


# For when a user wants a drawn mesh:
# The original image path is provided, from this a new filepath is calculated so that the saved file will use the same directory
# The annotated image is created, then, a mesh is drawn between all points over the image, and saved
def draw_with_points(image, results, path):
    clean_path = path.split(".")[0]  # prepare a path to save the annotated image to

    annotated_image = image.copy()

    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.multi_face_landmarks[0],
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())

    cv2.imwrite(clean_path + '_annotated.png', annotated_image)


def user_warning(message):  # A quick method to shortcut the generation of warnings elsewhere, and to keep code readability
    warnings.warn(message)
