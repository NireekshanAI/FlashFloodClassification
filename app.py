import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import torch
import time
import os
import plotly.graph_objects as go


st.set_page_config(page_title="Flash Flood Early Warning System", layout="wide")
st.title("Flash Flood Early Warning System")

@st.cache_resource
def load_model():
    return YOLO(rf"D:\Python Projects\yolov11\runs\classify\train4\weights\best.pt")

model = load_model()

upload,camera,live = st.tabs(["Upload",  "Camera", "Live"])
df = pd.DataFrame(columns=["Flash Flood Confidence", "Normal Flow Confidence", "Not Flash Flood Confidence"])


with upload:
    st.subheader("Upload an Image for Processing")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            campic = Image.open(uploaded_file)
            st.image(campic, caption="Uploaded Image", use_container_width=True)
            results = model.predict(source=campic)
            with col2:
                for result in results:
                    st.write("")
                    st.text("")
                    st.write("")
                    st.text("")
                    st.write("")
                    st.text("")
                    st.write("")
                    st.text("")
                    st.write("")
                    st.text("")
                    st.write("")
                    st.image(result.plot(), caption='Processed Image', use_container_width =True,channels="BGR")

with camera:
    st.subheader("Capture Image from Camera")
    col1, col2 = st.columns(2)
    with col1:
        pic = st.camera_input(" ")
        if pic:
            pic = Image.open(pic)
            results = model.predict(source=pic)
            p1,p2,p3 = results[0].probs.data[0].item(),results[0].probs.data[1].item(),results[0].probs.data[2].item()
            with col2:
                for result in results:
                    st.write("")
                    st.write("")
                    st.image(result.plot(), caption='Processed Image', use_container_width =True,channels="BGR")
                    new_row = pd.DataFrame({"Flash Flood Confidence": [p1],"Normal Flow Confidence": [p2],"Not Flash Flood Confidence": [p3]})
                    df = pd.concat([df, new_row], ignore_index=True)
                    st.dataframe(df) 

with live:
    
    st.subheader("Live Video Processing")
    col1, col2 = st.columns(2)
    with col2:
        stdframe = st.empty()
    with col1:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        start_button = st.button("Start Capture")
        stop_button = st.button("Stop Video Capture")
        plot_placeholder = st.empty()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Flash Flood'))
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Normal Flow'))
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Not Flash Flood'))
        fig.update_layout(title='Confidence Values Over Time',xaxis_title='Time',yaxis_title='Confidence')

        while start_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture video!")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(source=frame)
            stframe.image([x.plot() for x in results][0], caption='Processed Image', use_container_width =True,channels="RGB")  # Display only processed video feed
            p1,p2,p3 = results[0].probs.data[0].item(),results[0].probs.data[1].item(),results[0].probs.data[2].item()
            with col2:
                new_row = pd.DataFrame({"Flash Flood Confidence": [p1],"Normal Flow Confidence": [p2],"Not Flash Flood Confidence": [p3]})
                df = pd.concat([df, new_row], ignore_index=True)
                stdframe.dataframe(df, use_container_width =True,height=660) 
                if stop_button:
                    break
                fig.data[0].x = list(range(len(df)))
                fig.data[0].y = df["Flash Flood Confidence"]
                fig.data[1].x = list(range(len(df)))
                fig.data[1].y = df["Normal Flow Confidence"]
                fig.data[2].x = list(range(len(df)))
                fig.data[2].y = df["Not Flash Flood Confidence"]
                
                plot_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.1)
    cap.release()
    # st.subheader("Live Video Processing")
    # cap = cv2.VideoCapture(0)
#     stframe = st.empty()
#     stop = st.button("Stop Video Capture",key="sgsdg",type="primary")
#     # data = st.button("data",key="data")
#     while not stop:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("Failed to capture video!")
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = model.predict(frame)

#         p1,p2,p3 = results[0].probs.data[0].item(),results[0].probs.data[1].item(),results[0].probs.data[2].item() 
#         new_row = pd.DataFrame({"Flash Flood Confidence": [p1],"Normal Flow Confidence": [p2],"Not Flash Flood Confidence": [p3]})
#         df = pd.concat([df, new_row], ignore_index=True)
#         st.image(results.plot(), channels="RGB")
#         st.dataframe(df)
#     cap.release()


# def process_image(image):
#     results = model.predict(source=image)
#     processed_results = []
#     p1,p2,p3 = results[0].probs.data[0].item(),results[0].probs.data[1].item(),results[0].probs.data[2].item()

#     for result in results:
#         if hasattr(result, 'probs'):
#             probs = result.probs
#             flash_flood_confidence, top5_confidence = get_confidence_value(probs, 0)
#         else:
#             flash_flood_confidence, top5_confidence = 0.0, [0.0, 0.0, 0.0, 0.0, 0.0]

#         # Get times for preprocessing, inference, and postprocessing
#         if hasattr(result, 'speed') and isinstance(result.speed, list) and len(result.speed) == 3:
#             preprocess_time = f"{result.speed[0]}ms"
#             inference_time = f"{result.speed[1]}ms"
#             postprocess_time = f"{result.speed[2]}ms"
#         else:
#             preprocess_time = inference_time = postprocess_time = "N/A"
        
#         processed_results.append({
#             "confidences": {
#                 "Flash Flood": flash_flood_confidence,
#                 "Normal Flow": top5_confidence[0],
#                 "Not Flash Flood": top5_confidence[1]
#             },
#             "times": {
#                 "Preprocess": preprocess_time,
#                 "Inference": inference_time,
#                 "Postprocess": postprocess_time
#             },
#             "shape": image.size if isinstance(image, Image.Image) else image.shape,
#             "processed_image": result.plot()  # Get the processed image from the result
#         })
    
#     return processed_results, p1, p2, p3

# def main():
    

#     elif mode == "Upload Image":
#         st.subheader("Upload an Image for Processing")
#         uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
        
#         if uploaded_file is not None:
#             image = Image.open(uploaded_file)
#             st.image(image, caption="Uploaded Image", use_container_width=True)
#             results = process_image(image)[0]
            
#             # Display results in tabular format
#             for result in results[0]:
#                 st.table(pd.DataFrame([{
#                     'Flash Flood Confidence': result['confidences']['Flash Flood'],
#                     'Normal Flow Confidence': result['confidences']['Normal Flow'],
#                     'Not Flash Flood Confidence': result['confidences']['Not Flash Flood'],
#                     'Preprocess Time': result['times']['Preprocess'],
#                     'Inference Time': result['times']['Inference'],
#                     'Postprocess Time': result['times']['Postprocess'],
#                     'Image Shape': result['shape'],
#                 }]))
#                 st.image(result["processed_image"], caption="Processed Image", use_container_width=True)

#     elif mode == "Camera Capture":
#         st.subheader("Capture Image from Camera")
#         picture = st.camera_input("Take a picture")
        
#         if picture is not None:
#             image = Image.open(picture)
#             results = process_image(image)
            
#             # Display results in tabular format
#             st.table(pd.DataFrame([{
#                 'Flash Flood Confidence': results[1],
#                 'Normal Flow Confidence': results[2],
#                 'Not Flash Flood Confidence': results[3],
#                 # 'Preprocess Time': result['times']['Preprocess'],
#                 # 'Inference Time': result['times']['Inference'],
#                 # 'Postprocess Time': result['times']['Postprocess'],
#                 # 'Image Shape': result['shape'],
#             }]))
#             # st.image(results["processed_image"], caption="Processed Image", use_container_width=True)

#     # Save results to CSV
#     if 'results' in locals():
#         df = pd.DataFrame([{
#             'Flash Flood Confidence': results[1],
#             'Normal Flow Confidence': results[2],
#             'Not Flash Flood Confidence': results[3],
#             # 'Preprocess Time': results[0]['times']['Preprocess'],
#             # 'Inference Time': results[0]['times']['Inference'],
#             # 'Postprocess Time': results[0]['times']['Postprocess'],
#             # 'Image Shape': results[0]['shape'],
#             # 'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
#         } for result in results[0]])
#         df.to_csv(csv_file_path, mode='a', header=False, index=False)
#         st.subheader("Post-Processing Results")
#         st.write(df)
        
#         csv = df.to_csv(index=False)
#         st.download_button(
#             label="Download Post-Processing Results as CSV",
#             data=csv,
#             file_name="post_process_results.csv",
#             mime="text/csv"
#         )

# if __name__ == "__main__":
#     main()
