import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import base64
import os
from PIL import Image
import cv2
import numpy as np
import pytesseract


def load_image(image_file):
    img = Image.open(image_file)
    return img
  
st.set_page_config(initial_sidebar_state="expanded")

st.title("extract content from scanned images")
image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if image_file is not None:
    file_details = {"file_name":image_file.name,"file_type":image_file.type,"file_size":image_file.size}
    #st.write(file_details)
    # st.image(load_image(image_file)) 
    bytes_data = image_file.read()
    base64_encoded_data = base64.b64encode(bytes_data).decode('utf-8')
    #st.text_area("Base64 Encoded Data", value=base64_encoded_data, height=300)
    #st.image(load_image(image_file))
    
if st.button("Enter"):
  
  file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
  nparr = np.frombuffer(bytes_data, np.uint8)
  #st.write(file_bytes)
  #st.write(nparr)
  original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
  if original_image is not None:
    #st.write("Image loaded successfully with OpenCV!")
    st.sidebar.image(original_image, channels="BGR", caption="Image loaded with OpenCV")
    
    display_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    #st.write("Grayscale Image success")
    st.sidebar.image(display_image, caption="Grayscale Image (OpenCV)")
    
  else:
    st.error("Failed to decode image with OpenCV. Please check file format.")
    
  img_height, img_width = original_image.shape[:2]
  
  thresh_image = cv2.adaptiveThreshold(display_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize=31, C=10) # Constant subtracted from the mean/weighted mean
  
  contours, hierarchy = cv2.findContours(thresh_image.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Compress points
  
  
  detected_objects = []
  min_aspect_ratio = 0.8  # Example: Look for objects wider than they are tall (e.g., ~2.0 for many license plates)
  max_aspect_ratio = 1.5  # Max aspect ratio
  min_area = 500 # Optional: Filter out very small noise contours
  extracted_count = 0
  
  
  for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_area:
      continue
    
    x, y, w, h = cv2.boundingRect(contour)
    if h > 0:
      aspect_ratio = float(w) / h
    else:
      continue
    
    if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
      cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
      detected_objects.append(contour)
      #print(f"Detected object with Aspect Ratio: {aspect_ratio:.2f}, Area: {area}")
      extracted_object = original_image[y:y+h, x:x+w]
      #st.write(f"objects {extracted_count}")
      #st.image(extracted_object)
      extracted_count += 1
    
  result = pytesseract.image_to_string(Image.open(image_file))
  st.text_area("Text Data", value=result, height=100)
  
  gray_blurred = cv2.medianBlur(display_image, 5)
  output_image = display_image.copy()
  
  circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,  param1=50, param2=30, minRadius=50, maxRadius=80)
  
  if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
      center = (i[0], i[1])
      radius = i[2]
      cv2.circle(output_image, center, radius, (0, 255, 0), 2)
      cv2.circle(output_image, center, 2, (0, 0, 255), 3)
      
      padding = 5
      x_start = int(center[0] - radius - padding)
      y_start = int(center[1] - radius - padding)
      x_end = int(center[0] + radius + padding)
      y_end = int(center[1] + radius + padding)
      
      x_start = max(0, x_start)
      y_start = max(0, y_start)
      x_end = min(img_width, x_end)
      y_end = min(img_height, y_end)
      
      if x_end <= x_start or y_end <= y_start:
        print(f"Skipping circle {i}: Invalid bounding box dimensions ({x_start},{y_start}) to ({x_end},{y_end}).")
        continue
        
      cropped_circle_img = original_image[y_start:y_end, x_start:x_end]
      st.write(f"stamp{i}")
      st.image(cropped_circle_img)
      
      
    st.write(f"Found {len(circles[0])} circles.")
  else:
    st.write("No circles found.")
    
  #st.image(output_image)
  
  
      

  
  
  
