import os
import sys
import cv2
import numpy as np

def get_sorted_contours(img):
    contour_image = np.ones_like(img) * 255 #make new, all white image with size of "img". This is background for contours
    sorted_contour_image = np.ones_like(img) * 255
    #show_image(contour_image, 2000)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(contour_image, contours, -1, (0, 0, 0), 2)  # -1 means drawing all contours
    show_image(contour_image, 2000)

    contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    #contour visualization
    for contour in contours_sorted[0 :20]:
        cv2.drawContours(sorted_contour_image, [contour], -1, (0, 0, 255), 2)
        show_image(sorted_contour_image, 100)
    
    return contours_sorted

def get_array_of_images_of_potential_license_plates_by_contours(contours_sorted, img):
    array_of_images_of_potential_license_plates = []
    for rect in contours_sorted:
        x, y, w, h = cv2.boundingRect(rect) # Returns the coordinates and dimensions of the bounding rectangle around the contour

        if w in range(150, 590) and h in range(0, 150) and 2.5 <= w/h <= 6.5:
            #print(w/h)
            cropped_image = img[y:y+h, x:x+w]  # cutting image to get just license plate (and things that looks similar)
            show_image(cropped_image, 2000)

            array_of_images_of_potential_license_plates.append(cropped_image)

    return array_of_images_of_potential_license_plates

def get_array_of_images_of_potential_license_plates(img_original_resized, threshold_value):

    img_gray = cv2.cvtColor(img_original_resized, cv2.COLOR_BGR2GRAY)
    show_image(img_gray, 2000)
    img = cv2.bilateralFilter(img_gray, 11, 17, 17) 
    show_image(img, 2000)
    ret, img = cv2.threshold(img, threshold_value, 255, 0)
    show_image(img, 2000)
    contours_sorted = get_sorted_contours(img)

    array_of_images_of_potential_license_plates = get_array_of_images_of_potential_license_plates_by_contours(contours_sorted, img_gray)

    return array_of_images_of_potential_license_plates


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
    
    return array_of_images_of_potential_license_plates

def license_plate_detector():
    max_threshold = 180  
    min_threshold = 50  
    threshold = 130
    image_path= 'license_plates/DW3L111.jpg'
    result = read_image(max_threshold, min_threshold, threshold, image_path)


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
    license_plate_detector()

