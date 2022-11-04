# Optical-Character-Recogntion-Scanner-App
OCR Scanner Application with Streamlit, FastAPI, Tesseract and MongoDB

Sequential app functionality:
- User uploads image that should be scanned
- Image scanning (resizing, filtering, warping and cropping image) using OpenCV
- Detecting and recognizing characters using Tesseract
- Storing result to MongoDB and return it to user

### OCR app demo

[Scanner.webm](https://user-images.githubusercontent.com/31970304/200011832-028027c5-34d3-4643-b6e7-51fe8a66789f.webm)


### Arcitecture diagram
CLient-server 

![diagram](https://user-images.githubusercontent.com/31970304/200011963-513a8f9e-5d5b-412f-b883-5d3a45fc61d5.png)


### Scanning process with OpenCV `server/scan.py`
1. Resize initial image
2. Image filtering and edge detection
    - Gaussian and median blur
    - Canny edge detector
    - Erosion and dilation
3. Search of page's contour
4. Warp - image perspective transformation with page's contour that was found

<img src="https://user-images.githubusercontent.com/31970304/200015159-336bb0ac-0858-40e0-9009-8616de9b7592.jpg" height="300">
