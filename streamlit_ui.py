import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import image_processing
import time
import base64

st.set_page_config(page_title='Scanner', layout="wide", initial_sidebar_state='auto')


@st.cache(allow_output_mutation=True)
def load_image(upl_file):
    """
    Transform uploaded file to np array and resize
    :param upl_file: uploaded file
    :return: resized image
    """
    img = image_processing.resize(np.array(Image.open(upl_file)), 1200, 1200)
    return img


def preprocess_ui() -> tuple:
    st.sidebar.header('Process parameters')
    median_blur_param = st.sidebar.slider('Median Blur', 0, 50, 15)
    canny_param = st.sidebar.slider('Canny', 10, 200, 40)

    return median_blur_param, canny_param


def contours_ui() -> int:
    st.sidebar.header('Contour parameter')
    contour_param = st.sidebar.slider('Appriximation', 0.001, 0.3, 0.1)

    return contour_param


def crop_ui() -> int:
    st.sidebar.header('Crop parameter')
    crop_param = st.sidebar.slider('Crop image', 0, 100, 7)

    return crop_param


def scan_image(img):
    """
    Perform 'scanning' process
    :param img: default image
    :return: scanned img
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc_image = image_processing.preprocess(img_gray, median_blur_param, canny_param)
    contour = image_processing.get_contours(proc_image, img, contour_param)
    warped_img = image_processing.getWarp(img_gray, contour, crop_param)

    return warped_img


##########################################################################

st.title('CV SCanner')
st.header('Demo page')

st.sidebar.title('Set params')
median_blur_param, canny_param = preprocess_ui()
st.sidebar.write(median_blur_param, canny_param)

contour_param = contours_ui()
st.sidebar.write(contour_param)

crop_param = crop_ui()
st.sidebar.write(crop_param)



col1, col2, col3= st.columns([3, 1, 3])

with col1:
    st.subheader('Upload image')
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        img = load_image(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width='auto')

with col3:
    st.subheader('Scanning result')
    if uploaded_file is not None:
        try:
            scanned_img = scan_image(img)
            st.image(scanned_img, use_column_width='auto')
            # img_to_download = Image.fromarray(scanned_img)
            # img_to_download = base64.b64encode(img_to_download.encode()).decode()
            # st.download_button(
            #     label="Download scanned image",
            #     data=img_to_download,
            #     mime='image/png')
        except Exception as ex:
            print(ex)
            st.error('Unable to scan')

cont = st.container()
with cont:
    st.subheader('OCR Tesseract result')


#
# with col3:
#     st.subheader('Result')
#     cont3 = st.container()
#     with cont3:
#         with st.spinner('Wait for it...'):
#             time.sleep(5)
#         st.success('Done!')
#     if uploaded_file is not None:
#         st.image(image, caption='Uploaded Image.', use_column_width='auto')

# df = pd.read_csv(r'C:\Users\adima\Desktop\projects\python\HousePrice\data\test.csv')
# df2 = df.groupby(['MSZoning']).mean()['MasVnrArea']
#
#
# fig1 = px.line(df, x='Id', y="MasVnrArea")
# fig2 = px.histogram(df, x='MSZoning', y='MasVnrArea', barmode='group', histfunc='avg')
#
# st.set_page_config(page_title='Scanner', layout="wide", initial_sidebar_state='auto')
#
# st.title('CV SCanner')
# st.header('Demo page')
#
# st.write('Dataframe')
# st.dataframe(df)
#
# col1, col2= st.columns([1, 1])
#
# with col1:
#     st.write('Col1')
#     st.plotly_chart(fig1)
#
# with col2:
#     st.write('Col2')
#     st.plotly_chart(fig2)
#
# st.subheader('Upload image')
#
# uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
#
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width='auto')

# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width='auto')