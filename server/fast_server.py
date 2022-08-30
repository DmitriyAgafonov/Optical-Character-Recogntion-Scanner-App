import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import StreamingResponse
import uvicorn
import nest_asyncio
from pydantic import BaseModel
import json
from numpy import asarray
import pytesseract
# import cv2
# import numpy as np

from scan import *


dir_name = "images_uploaded"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def from_stream_to_image(bytes_stream):
    bytes_stream.seek(0)  # Start the stream from the beginning (position zero)
    file_bytes = asarray(bytearray(bytes_stream.read()), dtype=np.uint8)  # Write the stream of bytes into a numpy array
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return image


app = FastAPI(title='Deploying a ML Model with FastAPI')


class ScanParameters(BaseModel):
    median_blur: int
    canny: int
    contour: float
    crop: int
    ocr_status: bool = False

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.get("/")
def home():
    return "Congratulations! Your API is working. Now head over to http://localhost:8000/docs."


@app.post("/scan")
def perform_scanning(data: ScanParameters = Body(...),
                     file: UploadFile = File(...)):

    # Get parameters
    params = data.dict()

    median_blur = params['median_blur']
    canny = params['canny']
    contour = params['contour']
    crop = params['crop']
    ocr_status = params['ocr_status']

    print(median_blur, canny, contour, crop, ocr_status)

    # Validate File
    filename = file.filename
    # print(filename)
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    # Transform raw image into cv2 image
    img_stream = io.BytesIO(file.file.read())   # Read image as a stream of bytes
    original_img = from_stream_to_image(img_stream)

    # Perform scanning
    try:
        scanned_img = scan_image(original_img, median_blur, canny, contour, crop)
        # print(scanned_img)
        # print(type(scanned_img))

    except Exception as e:
        # print(f'except {e}')
        raise HTTPException(status_code=422, detail="Error while scanning image")

    cv2.imwrite(f'images_uploaded/{filename}', scanned_img)     # Save it in a folder

    # Response to client
    file_image = open(f'images_uploaded/{filename}', mode="rb")     # open image in binary

    # ocr_content = None
    # print(ocr_status)
    if ocr_status:
        # ocr_content = pytesseract.image_to_string(scanned_img)
        # print(ocr_content)
        return {'ocr_content': str(pytesseract.image_to_string(scanned_img))}
    else:
        # Return the image as a stream specifying media type
        return StreamingResponse(file_image, media_type="image/jpeg")


# nest_asyncio.apply()

# if __name__ == '__main__':
#     host = "127.0.0.2" if os.getenv("DOCKER-SETUP") else "localhost"  # or 'localhost'
#     uvicorn.run(app, host=host, port=8800, debug=True)
