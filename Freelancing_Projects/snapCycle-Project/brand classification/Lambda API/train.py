# from IPython.display import Image
from io import BytesIO
from PIL import Image
import urllib.request
import boto3, json
import numpy as np
import subprocess
import requests
import os


def lambda_handler(event, context):
    
    # DOWNLOAD IMAGE USING URL & SAVE INTO TMP FOLDER
    url_pet = 'https://d3r2zleywq7959.cloudfront.net/media/catalog/product/cache/1/image/9df78eab33525d08d6e5fb8d27136e95/2/2/22454_xlarge.jpg'  # pet image
    response = requests.get(url_pet)
    img1 = Image.open(BytesIO(response.content))
    img1.save('/tmp/test.jpg')
    print(os.listdir('/tmp/'))
    
    file_name = '/tmp/test.jpg'
    
    # ACCESS TO THE SAGEMAKER ENDPOINT
    ENDPOINT_NAME = os.environ['ENDPOINT']              # SAGEMAKER ENDPOINT connects to the environment variable
    runtime = boto3.Session().client(service_name='runtime.sagemaker')
    
    # READ IMAGE  AND IMAGE PROCESSING
    with open(file_name, 'rb') as f:
        payload = f.read()
        payload = bytearray(payload)
    
    # PREDICTION
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
                                      ContentType='application/x-image', 
                                      Body=payload)
    
    
    # RETURN THE PREDECTED RESULT
    result = response['Body'].read()
    result = json.loads(result)
    index = np.argmax(result)
    
    # PREDICTED RESULT CONVERT INTO CLASS LABELS
    if index == 0:
        return print('cocacola')
    elif index == 1:
        return print('lactania')
    elif index == 2:
        return print('ordinary')
    elif index == 3:
        return print('pepsi')
    elif index == 4:
        return print('starbucks')
    else:
        return print('image does not access ')
    
    # print(a)
    # return print('done')
