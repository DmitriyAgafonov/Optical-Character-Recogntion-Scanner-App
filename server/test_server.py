import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Body
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from typing import Optional, List
import json
from pydantic import BaseModel
import os
from numpy import asarray
import cv2
import numpy as np


def from_stream_to_image(bytes_stream):
    bytes_stream.seek(0)  # Start the stream from the beginning (position zero)
    file_bytes = asarray(bytearray(bytes_stream.read()), dtype=np.uint8)  # Write the stream of bytes into a numpy array
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return image


app = FastAPI(title='Deploying a ML Model with FastAPI')


@app.get("/")
def home():
    return "Congratulations! Your API is working. Now head over to http://localhost:8000/docs."


class Base(BaseModel):
    name: str
    point: Optional[float] = None
    is_accepted: Optional[bool] = False

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/test")
def submit(data: Base = Body(...), file: UploadFile = File(...)):
    print(data.dict())
    print(file.filename)
    print(file.content_type)
    filename = file.filename
    # return {"JSON Payload ": data, "Uploaded Files": file.filename}
    img_stream = io.BytesIO(file.file.read())  # Read image as a stream of bytes
    original_img = from_stream_to_image(img_stream)

    # Perform scanning
    scanned_img = cv2.bitwise_not(original_img)

    cv2.imwrite(f'images_uploaded/test{filename}', scanned_img)  # Save it in a folder

    # Response to client
    file_image = open(f'images_uploaded/test{filename}', mode="rb")  # open image in binary

    # ocr_response('qwertyuio')

    # Return the image as a stream specifying media type
    return StreamingResponse(file_image, media_type="image/jpeg", headers={'data': 'test'})


if __name__ == '__main__':
    host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "localhost"  # or 'localhost'127.0.0.1
    uvicorn.run(app, host=host, port=8001, debug=True)