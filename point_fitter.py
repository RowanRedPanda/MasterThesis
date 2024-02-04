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
    # Put a few comments here to describe what the program is looking for
    while True:
        user_input = input("Enter the path of your image: ")
        if not os.path.exists(user_input):
            print("File not found at " + str(user_input))
            continue
        else:
            path = user_input
            break
    while True:
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
    return fit_mesh(draw_mesh, path)


def fit_multiple_images():
    # user inputs
    # Put a few comments here to describe what the program is looking for
    while True:
        user_input = input("Enter the path of your file: ")
        if not os.path.exists(user_input):
            print("File not found at " + str(user_input))
            continue
        else:
            path = user_input
            break
    while True:
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
    i = 0
    j = 0
    measurements_list = []
    # path generation
    # What are we doing here?
    for filename in os.listdir(path):
        print(filename)
        i += 1
        image_path = path + '/' + filename
        landmarks = fit_mesh(draw_mesh, image_path)
        if bool(landmarks):
            # method to make measurements
            returned_dict = calculate_distances.calculate(landmarks)
            returned_dict["ImageName"] = filename
            measurements_list.extend([returned_dict])
            j += 1

    print(str(j) + ' images found and processed out of a total of ' + str(i) + ' files found')
    return measurements_list


def fit_mesh(draw_mesh, path):
    # set the parameters
    # What do these parameters mean/do?
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        # open the image
        image = cv2.imread(path)
        # get the image dimensions
        h, w, _ = image.shape

        # apply the mesh
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # set a bool on whether the results are invalid (at this stage a valid result is just one that contains some data)
    face_found = bool(results.multi_face_landmarks)

    if face_found:  # if something has been drawn, move to calculate, and possibly draw points

        if draw_mesh:  # if the end user wants a digital representation of the points mapped to the face
            draw_with_points(image, results, path)

        landmarks = results.multi_face_landmarks[0].landmark  # create a list of each x, y, z tuple

        for landmark in landmarks:
            # multiply out the results of the co-ordinates by the width and height to un-normalize them
            landmark.x *= w
            landmark.y *= h

        return results.multi_face_landmarks[0].landmark  # return the results to main

    else:  # if no face is found, return warning
        user_warning('Failed to find a face at ' + path)


# Briefly describe what this method is doing
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


def user_warning(message):  # generate warnings
    warnings.warn(message)
