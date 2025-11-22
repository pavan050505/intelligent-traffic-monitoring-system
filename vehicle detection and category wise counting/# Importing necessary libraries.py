# Importing necessary libraries
import streamlit as st
import cv2
from your_detection_script import process_video  # type: ignore # Import your process_video function from the backend code

st.title("Vehicle Detection and Category-wise Counting")

st.write("Upload a video to start vehicle detection and counting:")

# File upload widget
video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

# Process the uploaded video file
if video_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "/tmp/temp_video.mp4"  # Customize path as necessary
    with open(temp_file_path, "wb") as f:
        f.write(video_file.read())

    st.write("Processing the video, please wait...")

    # Run the detection function
    process_video(temp_file_path)

    # Display the resulting video
    st.video(temp_file_path)

