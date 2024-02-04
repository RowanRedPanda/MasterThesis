import warnings
import os
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic


def fit_points():  # fit a mesh to the face
    # user inputs
    # Probably just explain that this is a basic menu
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

    # set the parameters
    # What do these parameters mean?
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
            # multiply out the results of the co-ordinates by the width and height to de-normalize them
            landmark.x *= w
            landmark.y *= h

        return results.multi_face_landmarks[0].landmark  # return the results to main

    else:  # if no face is found, return warning
        user_warning('Failed to find a face')


# Summarise what this method does
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
