import pytesseract
import cv2

# read the image
original_image = cv2.imread(r'data\warped_img.png')
# preprocess the image using preprocess()
# preprocess() can be implemented using the cv2 methods medianBlur(), morphologyEx(), & threshold()
# preprocessed_image = preprocess(original_image)
# display text recognised in the preprocessed image
print(pytesseract.image_to_string(original_image))




