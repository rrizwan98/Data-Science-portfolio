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
    url_pet = 'https://www.modernretail.co/wp-content/uploads/sites/5/2022/07/pepsi-zero-sugar-rpet-bottle_3374x4106.jpeg'  # pet image
    response = requests.get(url_pet)
    img1 = Image.open(BytesIO(response.content))
    img1.save('/tmp/test.jpg')
    print(os.listdir('/tmp/'))
    
    file_name = '/tmp/test.jpg'
    
    # ACCESS TO THE SAGEMAKER ENDPOINT
    ENDPOINT_NAME = os.environ['ENDPOINT']         # SAGEMAKER ENDPOINT connects to the envionment variable
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
        return print('PET')
    elif index == 1:
        return print('HDPE')
    else:
        return print('others')
    
    # print(a)
    # return print('done')
