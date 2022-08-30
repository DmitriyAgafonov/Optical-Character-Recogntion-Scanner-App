import requests
import json
import streamlit as st


st.header('Test page')

url = 'http://localhost:8001/test'

# file = [('file', open('test_files/img1_small.jpg', 'rb'))]
# files = ('file', open(r'C:\Users\adima\Desktop\projects\python\doc_ocr_scanner\data\free-letter-mockup.jpg', 'rb'))

image = st.file_uploader("Choose an image")

data_dict = {"name": "foo", "point": 0.01, "is_accepted": False}
json_str = json.dumps(data_dict)
data = {'data': json_str}

if st.button("Send image"):
    if image is not None and data is not None:
        # file = {'file': image.getvalue()}
        file = {'file': image}
        resp = requests.post(url=url, data=data, files=file)
        # print(resp.json())
        if resp is not None:
            # print(resp.headers['data'])
            print(resp.content)
