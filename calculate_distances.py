import numpy

relevant_points = [
    (234, "Leftmost"),
    (454, "Rightmost"),
    (10, "Top"),
    (152, "Bottom"),
    (133, "L_I_I_Eye"),
    (362, "R_I_I_Eye"),
    (243, "L_I_E_Eye"),
    (463, "R_I_E_Eye"),
    (33, "L_O_I_Eye"),
    (263, "R_O_I_Eye"),
    (130, "L_O_E_Eye"),
    (359, "R_O_E_Eye"),
    (8, "Lower_Brow"),
    (9, "Upper_Brow"),
    (6, "Nose_Top"),
    (1, "Nose_Tip"),
    (102, "Left_Nose"),
    (331, "Right_Nose"),
    (2, "Nose_Base"),
    (0, "Mouth_Top"),
    (17, "Mouth_Bottom"),
    (61, "Left_Mouth"),
    (291, "Right_Mouth"),
    (175, "Chin_Tip"),
    (58, "Left_Jaw"),
    (288, "Right_Jaw"),
    (54, "Left_Forehead"),
    (284, "Right_Forehead")]
landmark_dict = {}
measurement_dict = {}


def calculate(landmarks):
    for point in relevant_points:  # For each point in the list of relevant points, create a dictionary entry
        # use the relevant point int from point[0] to index landmarks, i,e point 284 is landmarks[284]
        # point[1] is the name of the point, point[0] is the int of the point
        landmark_dict[point[1]] = landmarks[point[0]].x, landmarks[point[0]].y, landmarks[point[0]].z
        # Dictionary entry follows {String: (x, y, z)} format

    # normalise all face measurements from the width of the face, all faces are treated as width 1
    norm = 1/dist_calc(landmark_dict["Leftmost"], landmark_dict["Rightmost"])

    # prepare measurement dictionary of all completed measurements
    measurement_dict["Height"] = dist_calc(landmark_dict["Top"], landmark_dict["Bottom"]) * norm
    measurement_dict["Width of left eye"] = dist_calc(landmark_dict["L_I_E_Eye"], landmark_dict["L_O_E_Eye"]) * norm
    measurement_dict["Width of right eye"] = dist_calc(landmark_dict["R_I_E_Eye"], landmark_dict["R_O_E_Eye"]) * norm
    measurement_dict["Width between eyes"] = dist_calc(landmark_dict["L_I_E_Eye"], landmark_dict["R_I_E_Eye"]) * norm
    measurement_dict["Brow thickness"] = dist_calc(landmark_dict["Lower_Brow"], landmark_dict["Upper_Brow"]) * norm
    measurement_dict["Nose length"] = dist_calc(landmark_dict["Nose_Top"], landmark_dict["Nose_Tip"]) * norm
    measurement_dict["Nose width"] = dist_calc(landmark_dict["Left_Nose"], landmark_dict["Right_Nose"]) * norm
    measurement_dict["Nose and mouth gap"] = dist_calc(landmark_dict["Nose_Base"], landmark_dict["Mouth_Top"]) * norm
    measurement_dict["Mouth thickness"] = dist_calc(landmark_dict["Mouth_Top"], landmark_dict["Mouth_Bottom"]) * norm
    measurement_dict["Mouth width"] = dist_calc(landmark_dict["Left_Mouth"], landmark_dict["Right_Mouth"]) * norm
    measurement_dict["Chin to mouth"] = dist_calc(landmark_dict["Mouth_Bottom"], landmark_dict["Chin_Tip"]) * norm
    measurement_dict["Forehead"] = dist_calc(landmark_dict["Upper_Brow"], landmark_dict["Top"]) * norm
    measurement_dict["Forehead width"] = dist_calc(landmark_dict["Left_Forehead"], landmark_dict["Right_Forehead"]) * norm
    measurement_dict["Jaw width"] = dist_calc(landmark_dict["Left_Jaw"], landmark_dict["Right_Jaw"]) * norm
    measurement_dict["Left eye depth"] = dist_calc(landmark_dict["L_I_E_Eye"], landmark_dict["L_I_I_Eye"]) * norm
    measurement_dict["Right eye depth"] = dist_calc(landmark_dict["R_I_E_Eye"], landmark_dict["R_I_I_Eye"]) * norm
    measurement_dict["Edge to left eye"] = dist_calc(landmark_dict["Leftmost"], landmark_dict["L_O_E_Eye"]) * norm
    measurement_dict["Edge to right eye"] = dist_calc(landmark_dict["Rightmost"], landmark_dict["R_O_E_Eye"]) * norm

    return measurement_dict


def dist_calc(landmark_a, landmark_b):
    a = numpy.array(landmark_a)  # convert tuples to numpy arrays
    b = numpy.array(landmark_b)
    return numpy.linalg.norm(a - b)
