import streamlit as st
import requests
import json
import numpy as np
import cv2
import io
import os
import logging
import pymongo


st.set_page_config(page_title='Scanner', layout="wide", initial_sidebar_state='auto')


def preprocess_ui() -> tuple:
    st.sidebar.header('Process parameters')
    median_blur_param = st.sidebar.slider('Median Blur : 15', 0, 50, 15)
    canny_param = st.sidebar.slider('Canny : 40', 10, 200, 40)

    return median_blur_param, canny_param


def contours_ui() -> float:
    st.sidebar.header('Contour parameter')
    contour_param = st.sidebar.slider('Approximation : 0.1', 0.001, 0.3, 0.1)

    return contour_param


def crop_ui() -> int:
    st.sidebar.header('Crop parameter')
    crop_param = st.sidebar.slider('Crop image : 7', 0, 100, 7)

    return crop_param


def request(data, file, server_url: str):

    response = requests.post(url=server_url, data=data, files=file)

    return response


def from_stream_to_image(bytes_stream):
    bytes_stream.seek(0)    # Start the stream from the beginning (position zero)
    file_bytes = np.asarray(bytearray(bytes_stream.read()), dtype=np.uint8)     # Write the stream of bytes into a numpy array
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    return image


########################################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(stream=sys.stdout)
# handler.setFormatter(logging.Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
# logger.addHandler(handler)

logger.debug('debug info')

########################################################################


st.title('CV Scanner')

st.sidebar.title('Set params')


median_blur_param, canny_param = preprocess_ui()
st.sidebar.write(median_blur_param, canny_param)

contour_param = contours_ui()
st.sidebar.write(contour_param)


crop_param = crop_ui()
st.sidebar.write(crop_param)


params_to_server = {
    'median_blur': median_blur_param,
    'canny': canny_param,
    'contour': contour_param,
    'crop': crop_param
}

db_config = {
    "username": "root",
    "password": "secret",
    "server": "mongo-cont",
}

if os.getenv('DOCKER_VAR'):
    url, connector = \
        'http://scan_service:8000/scan', \
        'mongodb://{}:{}@{}'.format(
                                db_config['username'],
                                db_config['password'],
                                db_config['service']
                            )   # "mongodb://root:secret@mongo-cont"
else:
    url, connector = \
    'http://localhost:8000/scan', 'mongodb://localhost:27017/'

logger.debug('using url %s', url)

response = None
ocr_result = None
checkbox = None
button = None

col1, col2, col3 = st.columns([3, 1, 3])
with col1:
    st.subheader('Upload image')
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        logger.info('image was uploaded')
        file_img = {'file': uploaded_file}
        # print(uploaded_file.type)
        # print(type(uploaded_file))
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width='auto')

        button = st.button('Scan')
        checkbox = st.checkbox('Apply OCR')

        button_to_db = st.button('Save image to DB')

    params_to_server['ocr_status'] = checkbox

    json_str = json.dumps(params_to_server)
    data = {'data': json_str}

    if button:
        if data is not None and uploaded_file is not None:
            if checkbox:
                try:
                    st.info('Image + OCR')
                    response = request(data, file_img, url)
                    logger.info('file was sent with ocr checkbox, status code: %s', response.status_code)
                except Exception as e:
                    logger.exception('request with ocr failed, error at %s', e)
                # print(response.json())
                with col3:
                    st.subheader('OCR Content')
                    if response is not None and response.status_code == 200:
                        st.write(response.json()['ocr_content'])
                        logger.info('got ocr content')
                    else:
                        st.write('Error while scanning image and OCR. Change parameters!')
                        logger.info('failed to perform ocr')

            else:
                try:
                    st.info('Image')
                    response = request(data, file_img, url)
                    print(response.status_code)
                    logger.info('file only was sent, status code %s', response.status_code)
                except Exception as e:
                    logger.exception('request failed, error at %s', e)

                with col3:
                    st.subheader('Scanning result')
                    st.write(' ')
                    if response is not None and response.status_code == 200:
                        image_stream = io.BytesIO(response.content)  # Read image as a stream of bytes
                        image = from_stream_to_image(image_stream)
                        st.image(image, caption='Ready image', use_column_width='auto')
                        logger.info('got scan content')
                    else:
                        st.write("Error while scanning image. Change parameters!")
                        logger.info('failed to perform scanning')

            # st.success('Done!')
        else:
            st.error('Data or File is empty')

    if button_to_db:
        client = pymongo.MongoClient(connector)
        db = client['scandb']
        files_coll = db['files_collection']

        test_dict = { "name": "John", "address": "Highway 228" }

        document_id = files_coll.insert_one(test_dict).inserted_id

        logger.debug('db names %s , document id %s',
                     client.list_database_names(),
                     document_id)




    # else:
    #     st.error('Load pic first')

# with col3:
#     st.subheader('Scanning result')
#     if response is not None:
#         st.image(image, caption='Ready image', use_column_width='auto')
#
# col4, col5 = st.columns([3, 3])
# with col4:
#     st.write(params_to_server)
#
# with col5:
#     st.subheader('OCR')
#     st.write(ocr_result)

