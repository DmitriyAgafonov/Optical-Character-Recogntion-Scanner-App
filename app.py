import numpy as np
import cv2

img_name = 'img1.png'
path = 'data/{}'.format(img_name)



###################################

def resizeImage(img, newHeight=None, newWidth=None):
    dim = None
    h, w = img.shape[:2]

    if newWidth is None and newHeight is None:
        return img

    if (newWidth is None) or (h > w):
        r = newHeight / float(h)
        dim = (int(w * r), newHeight)

    if (newHeight is None) or (h <= w):
        r = newWidth / float(w)
        dim = (newWidth, int(h * r))
    print('new shape: ', dim[1], dim[0])

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def preprocessImage1(img):
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.bilateralFilter(greyImg, 9, 75, 75)
    # imgBlur = cv2.GaussianBlur(greyImg, (5, 5), 3)

    imgCanny = cv2.Canny(imgBlur, 40, 40)

    kern = np.ones((3, 3))

    imgDilate = cv2.dilate(imgCanny, kernel=kern, iterations=2)
    imgErode = cv2.erode(imgDilate, kernel=kern, iterations=1)

    return imgErode


def preprocessImage2(img):
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.bilateralFilter(greyImg, 9, 75, 75)
    # img = cv2.GaussianBlur(greyImg, (15, 15), 5)

    # Create black and white image based on adaptive threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    #
    # # Median filter clears small details
    # img = cv2.medianBlur(img, 25)

    # Add black border in case that page is touching an image border
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])


    img = cv2.Canny(img, 50, 50)

    kern = np.ones((3, 3))

    imgDilate = cv2.dilate(img, kernel=kern, iterations=2)
    imgErode = cv2.erode(imgDilate, kernel=kern, iterations=1)


    return imgErode


def getContours(img):
    biggestContour, maxArea = None, None
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contArea = [cv2.contourArea(contour) for contour in contours]

    sortedContoursAndArea = [(c, a) for c, a in sorted(zip(contours, contArea),
                                                       key=lambda pair: pair[1],
                                                       reverse=True)]

    # sortedContours, sortedArea = list(zip(*sortedContoursAndArea))
    # print(sortedArea)

    for contour, area in sortedContoursAndArea:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if area >= 5000 and len(approx) == 4:
            biggestContour = approx
            maxArea = area
            break

    print('Biggest contour: \n', biggestContour, '\nArea: ', maxArea)
    cv2.drawContours(imageCount, biggestContour, -1, (255, 0, 0), 20)

    return biggestContour



# def getContours(img):
#     biggestContour = np.array([])
#     maxArea = 0
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area >= 5000:
#             # cv2.drawContours(imageCount, contour, -1, (255, 0, 0), 3)
#             perimeter = cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
#             if area > maxArea and len(approx) == 4:
#                 biggestContour = approx
#                 maxArea = area
#
#     print('Biggest contour: \n', biggestContour, '\nArea: ', maxArea)
#     cv2.drawContours(imageCount, biggestContour, -1, (255, 0, 0), 20)
#
#     return biggestContour


def reorderPoints(points):
    points = points.reshape((4, 2))
    pointsNew = np.zeros((4, 1, 2), np.int32)
    summa = points.sum(axis=1)
    pointsNew[0] = points[np.argmin(summa)]
    pointsNew[3] = points[np.argmax(summa)]

    diff = np.diff(points, axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]

    return pointsNew


def getWarp(img, biggestContour):
    biggestContourReord = reorderPoints(biggestContour)

    (tl, tr, br, bl) = biggestContourReord.reshape((4, 2))

    widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
    widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    maxHeight = max(int(heightA), int(heightB))

    points1 = np.float32(biggestContourReord)
    points2 = np.float32([[-7, -7], [maxWidth + 7, -7], [-7, maxHeight + 7], [maxWidth +7, maxHeight + 7]])

    matrix = cv2.getPerspectiveTransform(points1, points2)
    warpedImg = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))

    print(warpedImg.shape)

    return warpedImg


##################################

# if __name__ == '__main__':
img = cv2.imread(path)
cv2.imshow('Default img', img)

imgHeight = img.shape[0]
imgWidth = img.shape[1]
print('Image shape: ', imgHeight, imgWidth)

resImg = resizeImage(img, 1100, 1300)
cv2.imshow('Resized img', resImg)

imageCount = resImg.copy()
finalImg = preprocessImage2(resImg)
cv2.imshow('Processed image', finalImg)

getContours(finalImg)

biggestContour = getContours(finalImg)
cv2.imshow('Contour img', imageCount)
#
warpedImg = getWarp(resImg, biggestContour)
cv2.imshow('Warp img', warpedImg)


cv2.waitKey(0)
cv2.destroyAllWindows()
