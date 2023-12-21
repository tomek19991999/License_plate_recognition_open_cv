import os
import sys
import cv2
import numpy as np

def get_sorted_contours(img):
    contour_image = np.ones_like(img) * 255
    sorted_contour_image = np.ones_like(img) * 255

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(contour_image, contours, -1, (0, 0, 0), 2)  # -1 means drawing all contours
    show_image(contour_image, 100)

    contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    #contour visualization
    for contour in contours_sorted[0 :20]:
        cv2.drawContours(sorted_contour_image, [contour], -1, (0, 0, 255), 2)
        #show_image(sorted_contour_image, 100)
    
    return contours_sorted

def get_array_of_images_of_potential_license_plates_by_contours_more_restricted(contours_sorted, img):
    array_of_images_of_potential_license_plates = []
    for c in contours_sorted:
        # Calculate the perimeter of the outline
        perimeter = cv2.arcLength(c, True)
        
        # Zoom in on the outline with a polygon
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        
        # We check whether the contour has 4 sides
        if len(approx) == 4 or len(approx) == 5:
            x, y, w, h = cv2.boundingRect(c)  

            # Perspective transformation to achieve a perpendicular view
            if w in range(150, 590) and h in range(0, 150) and 2.5 <= w/h <= 6.5:
                #print(w/h)
                dst = np.float32([[0, 0], [1000, 0], [0, 200], [1000, 200]])
                src = np.float32([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])
                matrix = cv2.getPerspectiveTransform(src, dst)
                result = cv2.warpPerspective(img, matrix, (1000, 200))

                #show_image(result, 500)
                result = result[:, 10:-10] #delete 10 pixels from right and left, to delete frame
                array_of_images_of_potential_license_plates.append(result)

    return array_of_images_of_potential_license_plates

def get_array_of_images_of_potential_license_plates_by_contours_old_less_restricted(contours_sorted, img):
    array_of_images_of_potential_license_plates = []
    for rect in contours_sorted:
        x, y, w, h = cv2.boundingRect(rect) # Returns the coordinates and dimensions of the bounding rectangle around the contour

        # Perspective transformation to achieve a perpendicular view
        if w in range(150, 590) and h in range(0, 150) and 2.5 <= w/h <= 6.5:
            #print(w/h)
            dst = np.float32([[0, 0], [1000, 0], [0, 200], [1000, 200]])
            src = np.float32([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])
            matrix = cv2.getPerspectiveTransform(src, dst)
            result = cv2.warpPerspective(img, matrix, (1000, 200))

            show_image(result, 1000)
            array_of_images_of_potential_license_plates.append(result)

    return array_of_images_of_potential_license_plates

def get_array_of_images_of_potential_license_plates(img_original_resized, threshold_value):

    img_gray = cv2.cvtColor(img_original_resized, cv2.COLOR_BGR2GRAY)
    #show_image(img_gray, 500)
    img = cv2.bilateralFilter(img_gray, 11, 17, 17) 
    #show_image(img, 1000)
    ret, img = cv2.threshold(img, threshold_value, 255, 0)
    #show_image(img, 500)
    contours_sorted = get_sorted_contours(img)

    array_of_images_of_potential_license_plates = get_array_of_images_of_potential_license_plates_by_contours_old_less_restricted(contours_sorted, img_gray)

    return array_of_images_of_potential_license_plates

def make_sorted_array_of_potential_letters(all_countours, potential_license_plate_img):
    # if contour have letter size, draw rectangle on img
    array_of_potential_letters = []
    for one_countour in all_countours:
        (x, y, w, h) = cv2.boundingRect(one_countour)

        #draw_contours_on_white_pic(one_countour, potential_license_plate_img, 200)

        if 100 < h < 195: 
            draw_contours_on_white_pic(one_countour, potential_license_plate_img, 100)

            array_of_potential_letters += [one_countour] #place good countour in array
            cv2.rectangle(potential_license_plate_img, (x, y), (x + w, y + h), (0, 0, 0), 1) #draw rectagnle on contour
            cv2.drawContours(potential_license_plate_img, one_countour, 0, (255, 255, 0), 1)

            show_image(potential_license_plate_img, 100)

    #we sort the outlines of potential characters based on "x" coordinates from left to right
    return sorted(array_of_potential_letters, key=lambda cnt: cv2.boundingRect(cnt)[0]) 

def filter_characters(sorted_array_of_potential_letters):
    filtered_characters = []

    prev_x = 0
    #filter chars too close to each other
    for char in sorted_array_of_potential_letters:
        x, y, w, h = cv2.boundingRect(char) 
        if prev_x == 0:
            prev_x = x
            filtered_characters.append(char)
            continue
        if x - prev_x < 50: 
            prev_x = x
            continue
        prev_x = x
        filtered_characters.append(char)
    return filtered_characters

def calculate_MSE_and_match_chars(filtered_sorted_array_of_potential_letters, potential_license_plate_img):
    plates = ""
    for letter in filtered_sorted_array_of_potential_letters:
        differences = dict()  # Create an empty dictionary to store differences (errors) between the contour and reference character
        minimum_error = sys.maxsize  # Set the minimum error to the maximum possible value in the system. It will store the smallest difference (error)

        for filename in os.listdir("characters/"):
            x, y, w, h = cv2.boundingRect(letter)  # Extract contour data
            char_image = potential_license_plate_img[y:y + h, x:x + w]  # Extract character from the image
            reference = cv2.imread("characters/" + filename, cv2.IMREAD_GRAYSCALE)  # Load reference character

            # Get dimensions of the character from the image
            width = int(char_image.shape[1])
            height = int(char_image.shape[0])
            dimensions = (width, height)

            # Resize the reference character to match the size of the character from the image and apply thresholding to both characters
            reference = cv2.resize(reference, dimensions)
            _, reference = cv2.threshold(reference, 127, 255, cv2.THRESH_BINARY)
            _, char_image = cv2.threshold(char_image, 127, 255, cv2.THRESH_BINARY)

            # Convert images to float and calculate the Mean Squared Error (MSE) (|float-float|^2)/sum_of_pixel_values)
            error = np.sum((char_image.astype("float") - reference.astype("float")) ** 2)
            error /= float(char_image.shape[0] * char_image.shape[1])
            number_of_white_pixels = error

            if number_of_white_pixels < minimum_error:
                minimum_error = number_of_white_pixels
            differences.update({filename: number_of_white_pixels})  # Build a dictionary with error for each reference character

        matched_character = {i for i in differences if differences[i] == minimum_error}.pop().split(".")[0]  # Find the best-matching character and append it to the plates
        plates += matched_character

    return plates


def recognize_plate_text(potential_license_plate_img):
    all_countours, _ = cv2.findContours(potential_license_plate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    draw_contours_on_white_pic(all_countours, potential_license_plate_img, 1000)

    cv2.drawContours(potential_license_plate_img,all_countours, -1, (0, 255, 255), 3)

    sorted_array_of_potential_letters = make_sorted_array_of_potential_letters(all_countours, potential_license_plate_img)

    filtered_sorted_array_of_potential_letters = filter_characters(sorted_array_of_potential_letters)

    matched_characters = calculate_MSE_and_match_chars(filtered_sorted_array_of_potential_letters, potential_license_plate_img)

    return matched_characters

def verify_detection(image_filename, detected_characters):
    # example: 'license_plates/DW2ST52.jpg' -> 'DW2ST52')
    real_characters = image_filename.split('.')[0]
    real_characters = real_characters.split('/')[1]

    print(f'Real characters: {real_characters}, Detected characters: {detected_characters}')

    if real_characters == detected_characters:
        print("Detected correctly")
        print("")
        return 1
    else:
        print("Detected wrong")
        print("")
        return 0

def read_image(max_threshold, min_threshold, threshold, filename):
    img_original = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_original_resized = cv2.resize(img_original, (640, 360))
    img_gray = cv2.cvtColor(img_original_resized, cv2.COLOR_BGR2GRAY)

    # search for license plate. If not found, change threshold
    while threshold <= max_threshold:
        array_of_images_of_potential_license_plates = get_array_of_images_of_potential_license_plates(img_original_resized, threshold)
        if array_of_images_of_potential_license_plates:
            break  
        threshold += 5  
    
    if not array_of_images_of_potential_license_plates:
        threshold = 120
        while threshold >= min_threshold:
            array_of_images_of_potential_license_plates = get_array_of_images_of_potential_license_plates(img_original_resized, threshold)
            if array_of_images_of_potential_license_plates:
                break  
            threshold -= 5  

    if not array_of_images_of_potential_license_plates:
        #sys.exit("License plate not found")
        return 0

    # Loop through potential license plates until a valid result is found
    for potential_license_plate in array_of_images_of_potential_license_plates:
        _, potential_license_plate = cv2.threshold(potential_license_plate, 80, 255, cv2.THRESH_BINARY)
        show_image(potential_license_plate, 1000)

        matched_characters = recognize_plate_text(potential_license_plate)

        if len(matched_characters) >= 6:
            break  # Exit the loop if a valid result is found

    if len(matched_characters) < 6 or len(matched_characters) > 10:
        # Loop through potential license plates until a valid result is found
        for potential_license_plate in array_of_images_of_potential_license_plates:
            _, potential_license_plate = cv2.threshold(potential_license_plate, 120, 255, cv2.THRESH_BINARY)
            show_image(potential_license_plate, 1000)

            matched_characters = recognize_plate_text(potential_license_plate)

            if len(matched_characters) >= 6:
                break  # Exit the loop if a valid result is found

    if len(matched_characters) < 6 or len(matched_characters) > 10:
        #sys.exit("License plate not found")
        return 0
    
    return verify_detection(filename, matched_characters) # return 1 if correct, if not correct return 0

def main():
    loop_flag = False
    max_threshold = 180  
    min_threshold = 50  
    threshold = 130
    if loop_flag == False:
        image_path= 'license_plates/DW3L111.jpg'
        result = read_image(max_threshold, min_threshold, threshold, image_path)

    else:
        folder_path = 'license_plates/'
        image_files = os.listdir(folder_path)

        correct_count = 0  
        incorrect_count = 0  

        for image_file in image_files:
            # Pełna ścieżka do obrazu
            image_path = os.path.join(folder_path, image_file)
            print(image_path)
            result = read_image(max_threshold, min_threshold, threshold, image_path)

            if result == 1:
                correct_count += 1
            else:
                incorrect_count += 1

        print("correct: ",correct_count)
        print("incorrect: ", incorrect_count)
        print("correct percent: ",correct_count / (correct_count+incorrect_count) ,"%")



def show_image(img, time):
    i = 1
    cv2.imshow('license_plate', img)
    cv2.waitKey(time)

def draw_contours_on_white_pic(data_to_draw, img_with_good_size, time):
    i = 1
    img = np.ones_like(img_with_good_size) * 255
    cv2.drawContours(img,data_to_draw, -1, (0, 0, 255), 3) 
    show_image(img, time)

if __name__ == '__main__':
    main()

