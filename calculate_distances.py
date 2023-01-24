import math

relevant_points = [
    (234, "Leftmost"),
    (454, "Rightmost"),
    (10, "Top"),
    (152, "Bottom"),
    (133, "L_I_I_Eye"),
    (362, "R_I_I_Eye"),
    (243, "L_I_E_Eye"),
    (463, "R_I_E_Eye"),
    (35, "L_O_I_Eye"),
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
    (17, "Mouth Bottom"),
    (61, "Left_Mouth"),
    (291, "Right_Mouth"),
    (175, "Chin_Tip"),
    (58, "Left_Jaw"),
    (288, "Right_Jaw"),
    (54, "Left_Forehead"),
    (284, "Right_Forehead")]
landmark_dict = {}


def calculate(landmarks):
    for point in relevant_points:
        landmark_dict[point[1]] = (landmarks[point[0]].x, landmarks[point[0]].y, landmarks[point[0]].z)
    print(landmark_dict)
