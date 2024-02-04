import calculate_distances
import point_fitter

landmarks = point_fitter.fit_points()  # request points fitted
if bool(landmarks):  # check for a valid result

    measurement_dict = calculate_distances.calculate(landmarks)
    # Comment
    for entry in measurement_dict:
        print(str(entry) + ": " + str(measurement_dict[entry]))
