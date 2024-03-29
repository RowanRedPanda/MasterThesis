import pandas as pd

import calculate_distances
import point_fitter

# user inputs
while True:
    user_input = input("""Would you like to measure faces, or predict a face?
    Enter 0 for measurement and 1 for prediction:""")  # Prediction functionality has been removed, selecting 0 to measure a face is the only function this has
    user_input = int(user_input)
    if user_input != 0 and user_input != 1:
        print("Please enter 0 or 1")
        continue
    else:
        if user_input == 0:  # Measuring faces
            while True:
                user_input = input("""Would you like to measure a single face, or an entire folder?
    Enter 0 for single, and 1 for multiple:""")  # To accomodate bulk measurements, the user is prompted to select single or multiple
                user_input = int(user_input)
                if int(user_input) != 0 and int(user_input) != 1:
                    print("Please enter 0 or 1")
                    continue
                elif user_input == 0:  # Single face processing requested
                    landmarks = point_fitter.fit_points()  # request points fitted
                    if bool(landmarks):  # check for a valid result

                        measurement_dict = calculate_distances.calculate(landmarks)
                        for entry in measurement_dict:
                            print(str(entry) + ": " + str(measurement_dict[entry]))
                    break
                else:  # Multiple face processing requested
                    landmarks = point_fitter.fit_multiple_images()  # request bulk fitting
                    df = pd.DataFrame(landmarks)
                    df.to_csv("FullMeasurements.csv", index=False)  # For bulk measurements, the measurement data is saved as a csv, for access and reading
                    break
        else:  # Face prediction requested
            # This code has been deliberately removed in order to adhere to ethical restrictions required by the research project
            break


