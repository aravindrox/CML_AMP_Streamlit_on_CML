import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import base64
import os
from PIL import Image


# client = OpenAI()
def llmcall(query,base64_encoded_data):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # cwd = os.getcwd()
    # image_path =  cwd+"/src/pdfcon1/data/image.png"
    # image_path =  "/Users/XXXXX/pdfcon1/src/pdfcon1/data/image.png"
    # base64_image = encode_image(image_path)
    # base64_image = base64.b64encode(image_file).decode('utf-8')

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        # 'text': 'how many stamps and logos are available'
                        'text': query
                    },
                    {
                    'type': 'image_url',
                    'image_url': 
                    {
                        #    'url': 'https://site/image.jpg',
                        'url': f'data:image/png;base64,{base64_encoded_data}',
                    },

                    },
                ],
            }
        ],
        max_tokens=300
    )
    return response
# print(response.choices[0].message)

def encode_image(image_path):
    '''Encodes am image to base 64 string'''
    with open(image_path,"rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_image(image_file):
    img = Image.open(image_file)
    return img

st.title("Line of Credit")
image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if image_file is not None:
    file_details = {"file_name":image_file.name,"file_type":image_file.type,"file_size":image_file.size}
    st.write(file_details)
    # st.image(load_image(image_file))
    bytes_data = image_file.read()
    base64_encoded_data = base64.b64encode(bytes_data).decode('utf-8')
    st.text_area("Base64 Encoded Data", value=base64_encoded_data, height=300)
    # st.text_area("Base64 Encoded Data path", value=encode_image("/Users/aswarna/pdfcon1/src/pdfcon1/data/image.png"), height=300)
    st.image(load_image(image_file))
query = st.text_input("Query regarding the image",placeholder="How many stamps are present")
    # st.write(llmcall())
if st.button("Enter"):
    llm_response = llmcall(query,base64_encoded_data)
    st.write(llm_response.choices[0].message.content)
    st.write("completion tokens: ", llm_response.usage.completion_tokens)
    st.write("Prompt tokens: ", llm_response.usage.prompt_tokens)
    st.write("total token: ", llm_response.usage.total_tokens)
