import calculate_distances
import point_fitter

landmarks = point_fitter.fit_points()  # request points fitted
if bool(landmarks):  # check for a valid result
    calculate_distances.calculate(landmarks)
