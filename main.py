import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyttsx3
import pandas as pd
from playsound import playsound
import threading
import time
from collections import defaultdict
import pickle
from PIL import Image, ImageEnhance
import random

# Constants
KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
ATTENDANCE_FILE = 'attendance.csv'
ALERT_SOUND = 'alert.mp3'
ENCODINGS_CACHE = 'encodings_cache.pkl'
DEFAULT_TOLERANCE = 0.45
DEFAULT_FRAME_SKIP = 2
MAX_SAVED_IMAGES = 5
REGISTRATION_TIME_LIMIT = 15  # 15 seconds

# Setup directories
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

def initialize_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame(columns=['Name', 'Date', 'Time', 'Confidence']).to_csv(ATTENDANCE_FILE, index=False)
    else:
        # Ensure the file has the correct columns
        df = pd.read_csv(ATTENDANCE_FILE)
        required_columns = ['Name', 'Date', 'Time', 'Confidence']
        if not all(column in df.columns for column in required_columns):
            pd.DataFrame(columns=required_columns).to_csv(ATTENDANCE_FILE, index=False)

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

def save_encodings_cache(encodings, names):
    with open(ENCODINGS_CACHE, 'wb') as f:
        pickle.dump({'encodings': encodings, 'names': names}, f)

def load_encodings_cache():
    if os.path.exists(ENCODINGS_CACHE):
        with open(ENCODINGS_CACHE, 'rb') as f:
            return pickle.load(f)
    return None

def load_known_faces():
    cache = load_encodings_cache()
    if cache:
        return cache['encodings'], cache['names']

    encodings, names = [], []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(KNOWN_FACES_DIR, filename)
        name = filename.split('_')[0]
        try:
            image = face_recognition.load_image_file(path)
            locations = face_recognition.face_locations(image, model='hog')
            if locations:
                encoding = face_recognition.face_encodings(image, known_face_locations=locations)
                if encoding:
                    encodings.append(encoding[0])
                    names.append(name)
        except Exception as e:
            st.warning(f"Skipping {filename}: {str(e)}")

    if encodings:
        save_encodings_cache(encodings, names)
    return encodings, names

def mark_attendance(name, confidence):
    today = datetime.now().strftime('%Y-%m-%d')
    now = datetime.now().strftime('%H:%M:%S')
    try:
        if confidence < 0.7:
            st.warning("Please come into proper light for better recognition.")
            return False

        df = pd.read_csv(ATTENDANCE_FILE)
        if not df[(df['Name'] == name) & (df['Date'] == today)].empty:
            return False

        new_entry = pd.DataFrame([[name, today, now, f"{confidence:.2f}"]], columns=['Name', 'Date', 'Time', 'Confidence'])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

        st.success("Attendance marked successfully!")
        speak(f"Welcome {name}")
        if os.path.exists(ALERT_SOUND):
            threading.Thread(target=playsound, args=(ALERT_SOUND,), daemon=True).start()
        return True
    except Exception as e:
        st.error(f"Error saving attendance: {str(e)}")
        return False

def augment_face(img):
    if random.random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def register_face_via_camera(name):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    saved_images = 0
    start_time = time.time()

    st.info("📸 Capturing multiple high-quality faces... Please move slightly and look at the camera.")

    with st.spinner('Registration in progress...'):
        # Capture the center view first
        while saved_images < MAX_SAVED_IMAGES:
            elapsed_time = time.time() - start_time
            remaining_time = REGISTRATION_TIME_LIMIT - int(elapsed_time)

            if remaining_time <= 0:
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera!")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')

            frame_copy = frame.copy()
            cv2.putText(frame_copy, f"Images: {saved_images}/{MAX_SAVED_IMAGES}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame_copy, f"Time left: {remaining_time} sec", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            stframe.image(frame_copy, channels="BGR", use_container_width=True)

            if face_locations:
                top, right, bottom, left = face_locations[0]
                face = rgb_frame[top:bottom, left:right]
                if face.size == 0:
                    continue
                pil_face = Image.fromarray(face)
                pil_face = augment_face(pil_face)
                filename = f"{name}_center_{saved_images+1}.jpg"
                pil_face.save(os.path.join(KNOWN_FACES_DIR, filename))
                saved_images += 1
                time.sleep(0.5)

        if saved_images < MAX_SAVED_IMAGES:
            st.info("Now, please turn your head to the left.")
            saved_images = capture_side_view(saved_images, name, 'left', cap, stframe)

        if saved_images < MAX_SAVED_IMAGES:
            st.info("Now, please turn your head to the right.")
            saved_images = capture_side_view(saved_images, name, 'right', cap, stframe)

    cap.release()
    stframe.empty()

    if saved_images > 0:
        st.success(f"✅ Registration complete! Saved {saved_images} images for {name}.")
        if os.path.exists(ENCODINGS_CACHE):
            os.remove(ENCODINGS_CACHE)
    else:
        st.error("❌ No faces detected. Try again with better lighting.")

def capture_side_view(saved_images, name, side, cap, stframe):
    start_time = time.time()
    while saved_images < MAX_SAVED_IMAGES:
        elapsed_time = time.time() - start_time
        remaining_time = REGISTRATION_TIME_LIMIT - int(elapsed_time)

        if remaining_time <= 0:
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera!")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')

        frame_copy = frame.copy()
        cv2.putText(frame_copy, f"Images: {saved_images}/{MAX_SAVED_IMAGES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame_copy, f"Time left: {remaining_time} sec", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        stframe.image(frame_copy, channels="BGR", use_container_width=True)

        if face_locations:
            top, right, bottom, left = face_locations[0]
            face = rgb_frame[top:bottom, left:right]
            if face.size == 0:
                continue
            pil_face = Image.fromarray(face)
            pil_face = augment_face(pil_face)
            filename = f"{name}_{side}_{saved_images+1}.jpg"
            pil_face.save(os.path.join(KNOWN_FACES_DIR, filename))
            saved_images += 1
            time.sleep(0.5)

    return saved_images

# --------- UPDATED RECOGNIZE FACES FUNCTION ---------
def recognize_faces(tolerance=DEFAULT_TOLERANCE, frame_skip=DEFAULT_FRAME_SKIP):
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        st.error("No known faces found. Please register faces first.")
        return

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    if not cap.isOpened():
        st.error("Failed to open camera!")
        return

    if "recognition_running" not in st.session_state:
        st.session_state.recognition_running = False

    start_button = st.button("▶️ Start Recognition")
    stop_button = st.button("🛑 Stop Recognition")

    if start_button:
        st.session_state.recognition_running = True
        st.success("Recognition Started ✅ - Look into the Camera!")

    if stop_button:
        st.session_state.recognition_running = False
        st.success("Recognition Stopped 🛑.")

    recognized_today = defaultdict(bool)
    frame_count = 0

    while st.session_state.recognition_running:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera disconnected.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small_frame, model='hog')

        for (top, right, bottom, left) in face_locations:
            top *= 2; right *= 2; bottom *= 2; left *= 2
            encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
            if encoding:
                encoding = encoding[0]
                distances = face_recognition.face_distance(known_encodings, encoding)
                min_distance = np.min(distances)
                best_match_idx = np.argmin(distances)

                name = "Unknown"
                confidence = 1 - min_distance

                if min_distance < tolerance:
                    name = known_names[best_match_idx]
                    if not recognized_today[name]:
                        if mark_attendance(name, confidence):
                            recognized_today[name] = True

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name} ({confidence*100:.1f}%)" if name != "Unknown" else "Unknown"
                cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    stframe.empty()

# ------------------------------------------------------

def batch_register_faces():
    uploaded_files = st.file_uploader("Upload a folder of images for batch registration", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            name = uploaded_file.name.split('_')[0]
            image_path = os.path.join(KNOWN_FACES_DIR, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Registered {uploaded_file.name} for {name}")
        st.info("Batch registration complete!")

# --------- ROLE-BASED FUNCTIONALITY ---------
def admin_section():
    st.subheader("Admin Panel")
    menu = st.selectbox("Admin Actions", ["Register Face", "Batch Register", "Start Recognition", "Attendance Log", "Add Attendance", "Delete Attendance"])

    if menu == "Register Face":
        name = st.text_input("Enter your name to register", "")
        if st.button("Start Registration"):
            if name:
                register_face_via_camera(name)
            else:
                st.warning("Please enter a valid name for registration.")

    elif menu == "Batch Register":
        batch_register_faces()

    elif menu == "Start Recognition":
        tolerance = st.slider("Set tolerance (lower is stricter)", 0.3, 0.6, 0.45, 0.05)
        frame_skip = st.slider("Frame skip", 1, 5, 2)
        recognize_faces(tolerance, frame_skip)

    elif menu == "Attendance Log":
        st.subheader("Attendance Log")
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            st.dataframe(df)
        except Exception as e:
            st.warning(f"No attendance data found. {str(e)}")

    elif menu == "Add Attendance":
        add_attendance_entry()

    elif menu == "Delete Attendance":
        delete_attendance_entry()

def add_attendance_entry():
    st.subheader("Add Attendance Record")
    try:
        df = pd.read_csv(ATTENDANCE_FILE)

        name = st.text_input("Enter Name:")
        date = st.date_input("Select Date:")
        time = st.time_input("Select Time:")
        confidence = st.slider("Confidence (0.0 to 1.0):", 0.0, 1.0, 0.7)

        if st.button("Add Attendance"):
            new_entry = pd.DataFrame([[name, date.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S'), f"{confidence:.2f}"]], 
                                     columns=['Name', 'Date', 'Time', 'Confidence'])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            st.success("Attendance record added successfully!")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error: {str(e)}")

def delete_attendance_entry():
    st.subheader("Delete Attendance Record")
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)

        row_to_delete = st.selectbox("Select the row to delete:", df.index)
        if st.button("Delete Selected Row"):
            df = df.drop(index=row_to_delete).reset_index(drop=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            st.success("Selected row deleted successfully!")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error: {str(e)}")

def student_section():
    st.subheader("Student Panel")
    recognize_faces(tolerance=DEFAULT_TOLERANCE, frame_skip=DEFAULT_FRAME_SKIP)

def main():
    st.set_page_config(page_title="Advanced Face Recognition", layout="wide")
    st.title("👤 Face Recognition Attendance System")

    role = st.sidebar.radio("Select Role", ["Admin", "Student"])

    if role == "Admin":
        admin_section()
    elif role == "Student":
        student_section()

if __name__ == "__main__":
    initialize_attendance_file()
    main()
