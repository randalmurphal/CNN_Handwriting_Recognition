import cv2
import numpy as np

def resize(img, height=1500):
    """ Resize image to given height """
    rat = height / img.shape[0]
    return rat, cv2.resize(img, (int(rat * img.shape[1]), height))

def findCorners(img, sigma):
    """ return ndarray with contains the coordinates of document's 4 corners """
    # Resize the image for easier detection
    rat, resized_img = resize(img)
    # Grayscale the image
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # BilateralFilter the image
    fltrd_img = cv2.bilateralFilter(gray_img, 5, 20, 20)

    # Find the edges of the the filtered image
    fltrd_edged_img = autoCanny(fltrd_img, sigma)

    # Get the different contours of the image
    contours,_ = cv2.findContours(fltrd_edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_cnt = contours[0]
    max_area = cv2.contourArea(max_cnt)

    for cnt in contours:
        curr_area = cv2.contourArea(cnt)
        if curr_area > max_area:
            max_area = curr_area
            max_cnt = cnt

    # Approximate the polygon for the biggest contour
    epsilon = 0.1*cv2.arcLength(max_cnt,True)
    corners = cv2.approxPolyDP(max_cnt,epsilon,True)
    # Use the ratio saved from downsizing the image to upsize the polygon to the original images size
    scaled_corners = np.int0(corners/rat)

    return scaled_corners


def autoCanny(resized_img, sigma):
    """Simple method for canny edge detection without the need for setting thesholds"""
    # Auto canny courtesy of:
    # https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    # compute the median of the single channel pixel intensities
    v = np.median(resized_img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged_img = cv2.Canny(resized_img, lower, upper)
    return edged_img


def transformPerspective(img, corners):
    """Transform the perspective of the image so that it has the specified corners"""
    points = [point.tolist()[0] for point in corners]
    points.sort(key=lambda p: p[1])
    points = sorted(points[:2], key=lambda p: p[0]) + sorted(points[2:], key=lambda p: p[0])

    width = max([points[1][0] - points[0][0], points[3][0] - points[2][0]])
    height = max([points[2][1] - points[0][1], points[3][1] - points[1][1]])

    input_points = np.float32(points)
    output_points = np.float32([[0,0],[width,0],[0,height],[width,height]])

    transform = cv2.getPerspectiveTransform(input_points, output_points)
    return cv2.warpPerspective(img, transform, (width,height))


def getTopDownView(img, corners, sigma):
    """Perform image alignment"""
    warped = transformPerspective(img, corners)
    rat, resized_img = resize(warped)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    fltrd_img = cv2.bilateralFilter(gray_img, 5, 20, 20)
    fltrd_edged_img = autoCanny(fltrd_img, sigma)
    contours,_ = cv2.findContours(fltrd_edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the smallest area greater than the threshold
    min_cnt = None
    min_area = fltrd_edged_img.shape[0]*fltrd_edged_img.shape[1]
    area_thresh = 0.5*min_area

    for cnt in contours:
        curr_area = cv2.contourArea(cnt)
        if curr_area > area_thresh and curr_area < min_area:
            min_area = curr_area
            min_cnt = cnt

    if min_cnt is None: return warped

    # Approximate the polygon for the contour
    epsilon = 0.1*cv2.arcLength(min_cnt,True)
    new_corners = cv2.approxPolyDP(min_cnt,epsilon,True)

    scaled_corners = np.int0(new_corners/rat)
    return transformPerspective(warped, scaled_corners)

def pageRecognition(inputImage):
    corners = findCorners(inputImage, .5)
    # Warp the image based on the corners found
    warped = getTopDownView(inputImage, corners, .5)
    return warped

def main():
    gold = cv2.imread('../form-gold-image.jpg')
    img = cv2.imread('./test2.png')
    corners = findCorners(img, .5)

    # Draw the polygon on the original image
    cv2.drawContours(img, [corners], -1, (0,255,0), 3)
    cv2.imshow("img", img)

    # Warp the image based on the corners found
    warped = getTopDownView(img, corners, .5)
    cv2.imshow('warped', warped)
    cv2.waitKey(0)

    # Resize the warped image so that it will fit on the gold standard
    warped = cv2.resize(warped, (gold.shape[1], gold.shape[0]))

    # Show the overlay of the warped and resized image with the gold standard
    cv2.imshow('warped', warped + gold)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()