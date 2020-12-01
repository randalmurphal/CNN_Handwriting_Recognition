import cv2
import numpy as np
import shutil
import os


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)


def sort_boxes(boxes):
    # https://stackoverflow.com/questions/38805462/how-to-sort-contours-left-to-right-while-going-top-to-bottom-using-python-and

    # [x,y,w,h]
    # sort by y
    boxes.sort(key=lambda x: x[1])
    num_boxes = len(boxes)

    # bottom row of boxes
    bottom_line = boxes[0][1] + boxes[0][3] - 1
    index = 0
    for i in range(num_boxes):
        # x is greater than the bottom line, need a new line
        if boxes[i][1] > bottom_line:
            # sort the previous line by their x
            boxes[index:i] = sorted(boxes[index:i], key=lambda x: x[0])
            index = i

        # check if found a new bottom line
        bottom_line = max(boxes[i][1] + boxes[i][3] - 1, bottom_line)

    # sort the last line
    boxes[index:] = sorted(boxes[index:], key=lambda x: x[0])
    return boxes


def overlapping_contours(box_1, box_2):
    return all(abs(box_1[i] - box_2[i]) <= 6 for i in range(4))


def save_boxes(boxes, original):
    png_path = 'individual_boxes/'
    contours_path = 'contours/'
    create_directory(png_path)
    create_directory(contours_path)
    image_width = original.shape[0]
    image_height = original.shape[1]

    image_number = 1
    for square in boxes:
        x = square[0]
        y = square[1]
        w = square[2]
        h = square[3]

        # save actual square
        square_box = original[y:y + h, x:x + w]
        cv2.imwrite(png_path + 'square_box_{}.png'.format(image_number), square_box)

        # save contour for reference
        img_copy = original.copy()
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv2.rectangle(img_copy, p1, p2, color=(0), thickness=3)
        cv2.imwrite(contours_path + 'contours_{}.png'.format(image_number), img_copy)
        image_number += 1


def extract_boxes(boxes, original):
    output = []

    for square in boxes:
        x = square[0]
        y = square[1]
        w = square[2]
        h = square[3]

        # save actual square
        square_box = original[y:y + h, x:x + w]
        output.append(square_box)

    return output


def get_boxes(img):
    # # 1. conversion to gray scale
    # # 2. binarization of the image. Using thresholding
    # # read image and apply grayscale
    # img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    # create a copy to extract fields from it later
    img = cv2.resize(img, (1300, 1700))
    original = img.copy()
    # thresholding - simplest method of segmenting images (process of partitionning a digital image into multiple segments, goal is to simplify image into a representation that is easier to analyze)
    # takes in grescale image. Used to create a binary image, returns threshold image
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, threshold = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY)
    # mode cv2.RETR_TREE: way to find contours
    # method cv2.CHAIN_APPROX_SIMPLE: approximation method for the detection. contour line indicates a line representing a boundary of the same intensities
    # returned contours is a list of points consisting of the contour lines
    contours, _ = cv2.findContours(image=threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    min_box_area = 1200
    max_box_area = 1800
    image_number = 1
    rejects_number = 1
    accepted_boxes = []
    i = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        # reducing the number of points in a curve with a reduced set of points
        # curve = series of short line segments*cv2.arcLength(cnt, True)
        # approximated curve = subset of points in original curve
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        img_copy = original.copy()
        # draw an approximate rectangle around the contour
        x, y, w, h = cv2.boundingRect(cnt)
        if len(approx) == 4 and area > min_box_area and area < max_box_area:
            overlapping = False
            if any(overlapping_contours(i, [x, y, w, h]) for i in accepted_boxes):
                overlapping = True
            if not overlapping:
                accepted_boxes.append([x, y, w, h])
                i += 1

    accepted_boxes = sort_boxes(accepted_boxes)
    boxes = extract_boxes(accepted_boxes, original)

    return boxes

# image_name = 'scanned_image.png'
# img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
# boxes = get_boxes(img)
# cv2.imshow("img", boxes[10])
# cv2.waitKey(0)