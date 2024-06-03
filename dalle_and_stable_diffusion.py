import streamlit as st
import os
import time
import requests
from PIL import Image, UnidentifiedImageError
import io
from dotenv import load_dotenv
from openai import OpenAI
client = OpenAI()
openai_api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()
API_URL = os.getenv("API_URL")
api_key = os.getenv("API")
headers = {"Authorization": api_key}

def query_huggingface(payload, retries=3, backoff=2):
    for i in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response
        elif response.status_code == 503:
            time.sleep(backoff)
            backoff *= 2
        else:
            return response
    return response

def generate_image_openai(prompt):
    response =client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        quality="standard",
    )
    image_url = response.data[0].url
    image = Image.open(io.BytesIO(requests.get(image_url).content))
    return image

st.title("Image Generation App")
st.write("Select a method to generate an image and enter a text prompt:")

method = st.radio("Select Method", ("Hugging Face", "OpenAI DALL-E 3"))

prompt = st.text_input("Text Prompt", "Astronaut riding a horse")


if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            if method == "Hugging Face":
                response = query_huggingface({"inputs": prompt})
                
                if response.status_code == 200:
                    try:
                        image_bytes = response.content
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, caption=f"Generated Image for: {prompt} using Hugging Face")
                    except UnidentifiedImageError:
                        st.error("The response content is not a valid image.")
                elif response.status_code == 503:
                    st.error("Service is currently unavailable. Please try again later.")
                else:
                    st.error(f"Failed to generate image. Status code: {response.status_code}, Response: {response.text}")
            elif method == "OpenAI DALL-E 3":
                try:
                    image = generate_image_openai(prompt)
                    st.image(image, caption=f"Generated Image for: {prompt} using OpenAI DALL-E 3")
                except Exception as e:
                    st.error(f"Error generating image: {e}")
    else:
        st.error("Please enter a prompt.")
