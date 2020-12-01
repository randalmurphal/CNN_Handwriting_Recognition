import cv2


from CoreBackend import box_extraction as be
from CoreBackend import page_recognition as pr

def main():
    pass
    # import numpy as np
    #
    # image_name = 'test2.png'
    # boxes = process_image(image_name)
    # cv2.imshow('boxes', np.concatenate(boxes))
    # cv2.waitKey(0)

def process_image(image_path):
    img = cv2.imread(image_path)
    page = pr.pageRecognition(img)
    page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    boxes = be.get_boxes(page)
    boxes = [cv2.resize(box, (28,28)) for box in boxes]
    boxes = [box.reshape((28,28,1)) for box in boxes]

    return boxes

if __name__ == "__main__":
    main()