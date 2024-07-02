from django.shortcuts import render

# Create your views here.

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseServerError
import requests

import cv2
import face_recognition
import pickle
from pathlib import Path
import numpy as np
import time
import requests
from django.http import JsonResponse, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")  # Adjust path if needed

def send_data_to_esp32(data):
    """
    This function sends data to the ESP32 using a simple HTTP POST request.
    You'll need to adjust this based on your chosen communication method.
    """

    try:
        # Replace with the actual URL of your ESP32 web server


        ip_address = 'http://127.0.0.1:8000/face_rec/data'  # Replace with your ESP32 IP address
        # url = f'http://{ip_address}/data'
        # Construct the data to send
        payload = {'recognized_name': data}

        # Send the data to the ESP32 using a POST request
        response = requests.post(ip_address, json=payload)

        # Check for successful response
        if response.status_code == 200:
            print(f"Data sent successfully to ESP32: {data}")
        else:
            print(f"Error sending data to ESP32: {response.status_code}")

    except Exception as e:
        print(f"Error sending data to ESP32: {e}")



# Function to load known encodings
def load_known_encodings(encodings_location):
    encodings_location = Path(encodings_location)

    with encodings_location.open(mode='rb') as f:
        known_encodings = pickle.load(f)

    return known_encodings

# Function to compare unknown encoding with known encodings and return the result
def compare_unknown_encoding(unknown_encoding, known_encodings, threshold=0.5):
    best_match = None
    best_distance = float('inf')

    for name, encodings in known_encodings.items():
        for known_encoding in encodings:
            distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
            if distance < best_distance:
                best_match = name
                best_distance = distance

    if best_distance <= threshold:
        return best_match
    else:
        return "Unknown"

# Load known encodings once at the start
known_encodings = load_known_encodings(DEFAULT_ENCODINGS_PATH)

@csrf_exempt
def recognize_face_view(request):
    

    try:
        # Initialize video capture (assuming you're using a webcam)
        video_capture = cv2.VideoCapture(0)
        start_time = time.time()

        while True:
            # Read frame-by-frame from video capture
            ret, frame = video_capture.read()
            if not ret:  # If no frame is captured, break the loop
                break

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            unknown_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model="small")

            for unknown_encoding, (top, right, bottom, left) in zip(unknown_encodings, face_locations):
                # Compare the face encoding with known encodings
                result = compare_unknown_encoding(unknown_encoding, known_encodings, threshold=0.5)

                # Draw a box around the face
                cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.putText(small_frame, result, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

                # Perform action based on recognition result
                if result == "Unknown":
                    # Send data to ESP32 (we'll discuss communication methods below)
                    send_data_to_esp32(result)  # Replace with your communication function
                else:
                    # Send data to ESP32 (we'll discuss communication methods below)
                    send_data_to_esp32(result)  # Replace with your communication function

            # Display the resulting image with annotated faces (optional)
            #cv2.imshow('Video', small_frame)
            #cv2.waitKey(1)  # Keep the window open briefly for display

            # Break the loop if 'q' is pressed or if no more frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            end_time = time.time()
            time_diff = (end_time - start_time)
            fps = 1.0 / time_diff
            start_time = end_time

        # Release video capture and close all OpenCV windows
        video_capture.release()
        cv2.destroyAllWindows()

        return JsonResponse({'status': 'processing'})

    except Exception as e:
        return JsonResponse({'error': str(e)})

    return JsonResponse({'status': 'invalid request'})