
# app.py
# ============================================
# Author: Florentine MUKAMANA
# Email:  muflorentine3@gmail.com
# Date:   12-July-2022 (refactored for DeepFace, 2026)
# ============================================

import os
import io
import cv2
import glob
import pickle
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from datetime import datetime
from typing import Optional, Tuple, List

from deepface import DeepFace
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Paths / Artifacts
# ------------------------------
ARTIFACTS_DIR = "artifacts"
UPLOAD_DIR = os.path.join(ARTIFACTS_DIR, "uploads")
PICKLE_DIR = os.path.join(ARTIFACTS_DIR, "pickle_format")
FEATURES_DIR = os.path.join(ARTIFACTS_DIR, "feature_extraction")
FACES_DIR = os.path.join(ARTIFACTS_DIR, "faces")  # used for webcam captures
ASSETS_DIR = "assets"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PICKLE_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

IMG_FILENAMES_PKL = os.path.join(PICKLE_DIR, "img_filenames.pkl")
FEATURES_PKL      = os.path.join(FEATURES_DIR, "features.pkl")
HAAR_PATH         = os.path.join(ASSETS_DIR, "haarcascade_frontalface_default.xml")

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(
    page_title="Insurance Face Recognition (Rwanda)",
    layout="wide",
    page_icon="🧠"
)

# ------------------------------
# Utilities
# ------------------------------
@st.cache_resource
def get_detector():
    """Cache the MTCNN detector (heavy object)."""
    return MTCNN()

def markdownstreamlit(text: str, tag: str = "p"):
    """Render minimal HTML with safe styling."""
    st.markdown(
        f'<{tag} style="background:#800000;color:#fff;'
        f'font-size:20px;border-radius:8px;padding:6px 10px;">{text}</{tag}>',
        unsafe_allow_html=True
    )

def name_from_path(path: str) -> str:
    """Extract a display name from filename like data/John_Doe/img1.jpg -> 'John Doe'."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]          # e.g., John_Doe_001
    parts = stem.split('_')
    # Try to recover first two tokens as a name; fall back to stem
    return " ".join(parts[:2]) if len(parts) >= 2 else stem.replace('_', ' ')

def save_uploaded_image(uploaded_file) -> Optional[str]:
    """Save an uploaded file to UPLOAD_DIR and return full path."""
    if uploaded_file is None:
        return None
    dst = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst

def cv2_im_from_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to BGR numpy array."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def ensure_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def ensure_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ------------------------------
# DeepFace Embeddings
# ------------------------------
def extract_embedding_from_crop(face_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Get a VGG-Face embedding for a cropped RGB face.
    DeepFace.represent accepts numpy images; we skip detection (already cropped).
    """
    try:
        rep = DeepFace.represent(
            img_path=face_rgb,
            model_name="VGG-Face",
            detector_backend="skip",
            enforce_detection=False,  # don't throw if slightly off
            normalization="base"
        )
        return np.array(rep[0]["embedding"], dtype="float32")
    except Exception as e:
        st.error(f"Embedding extraction failed: {e}")
        return None

def extract_features(img_path: str, detector: MTCNN) -> Optional[np.ndarray]:
    """Detect face via MTCNN and return VGG-Face embedding."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = ensure_rgb(img_bgr)
    results = detector.detect_faces(img_rgb)
    if not results:
        return None
    x, y, w, h = results[0]["box"]
    x, y = max(0, x), max(0, y)
    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
    return extract_embedding_from_crop(face)

def recommend(feature_list: List[np.ndarray], vec: np.ndarray) -> Tuple[int, float]:
    """Return (best_index, best_score) via cosine similarity."""
    sims = [cosine_similarity(vec.reshape(1, -1), f.reshape(1, -1))[0, 0] for f in feature_list]
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])

# ------------------------------
# Feature (re)builder
# ------------------------------
def rebuild_features_from_data(detector: MTCNN) -> Tuple[List[str], List[np.ndarray]]:
    """
    Build embeddings from images under 'data/<person>/*.jpg|png|jpeg'.
    Save both filenames and embeddings to the pickles.
    """
    patterns = ["data/**/*.jpg", "data/**/*.jpeg", "data/**/*.png"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(files)

    if not files:
        st.warning("No images found in 'data/'. Add images under data/<person_name>/ and retry.")
        return [], []

    filenames, features = [], []
    pbar = st.progress(0)
    for i, fpath in enumerate(files, start=1):
        emb = extract_features(fpath, detector)
        if emb is not None:
            filenames.append(fpath)
            features.append(emb)
        pbar.progress(i / len(files))

    if filenames and features:
        pickle.dump(filenames, open(IMG_FILENAMES_PKL, "wb"))
        pickle.dump(features, open(FEATURES_PKL, "wb"))
        st.success(f"Rebuilt features for {len(filenames)} images.")
    else:
        st.error("Failed to build features—no valid embeddings produced.")

    return filenames, features

# ------------------------------
# Database helpers (sqlite3)
# Three separate DBs to mirror your original code.
# ------------------------------
DB1_PATH = "db_soras.sqlite3"
DB2_PATH = "db_prime.sqlite3"
DB3_PATH = "db_rssb.sqlite3"

def get_conn(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn, conn.cursor()

def init_tables():
    # db1 -> SORAS
    conn1, c1 = get_conn(DB1_PATH)
    c1.execute("""
        CREATE TABLE IF NOT EXISTS tasksTabledb1(
            id INTEGER PRIMARY KEY,
            name TEXT,
            address TEXT,
            age INTEGER,
            identity_numb TEXT,
            due_date TEXT,
            photo BLOB
        )
    """)
    conn1.commit()

    # db2 -> PRIME insurance
    conn2, c2 = get_conn(DB2_PATH)
    c2.execute("""
        CREATE TABLE IF NOT EXISTS tasksTabledb2(
            id INTEGER PRIMARY KEY,
            name TEXT,
            address TEXT,
            age INTEGER,
            identity_numb TEXT,
            due_date TEXT,
            photo BLOB
        )
    """)
    conn2.commit()

    # db3 -> RSSB
    conn3, c3 = get_conn(DB3_PATH)
    c3.execute("""
        CREATE TABLE IF NOT EXISTS tasksTable(
            id INTEGER PRIMARY KEY,
            name TEXT,
            address TEXT,
            age INTEGER,
            identity_numb TEXT,
            due_date TEXT,
            photo BLOB
        )
    """)
    conn3.commit()

def add_data(db_path: str, row: Tuple):
    conn, c = get_conn(db_path)
    c.execute("""
        INSERT INTO {table}(id, name, address, age, identity_numb, due_date, photo)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """.format(table=os.path.splitext(os.path.basename(db_path))[0]
               .replace("db_", "tasksTable")), row)
    conn.commit()

def add_data_named(db_path: str, id_, name, address, age, identity_numb, due_date, photo):
    # Map file name to table
    table = {
        DB1_PATH: "tasksTabledb1",
        DB2_PATH: "tasksTabledb2",
        DB3_PATH: "tasksTable"
    }[db_path]
    conn, c = get_conn(db_path)
    c.execute(f"""
        INSERT INTO {table}(id, name, address, age, identity_numb, due_date, photo)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (id_, name, address, age, identity_numb, due_date, photo))
    conn.commit()

def view_all_data(db_path: str) -> List[Tuple]:
    table = {
        DB1_PATH: "tasksTabledb1",
        DB2_PATH: "tasksTabledb2",
        DB3_PATH: "tasksTable"
    }[db_path]
    conn, c = get_conn(db_path)
    c.execute(f"SELECT * FROM {table}")
    return c.fetchall()

def view_unique_names(db_path: str) -> List[str]:
    table = {
        DB1_PATH: "tasksTabledb1",
        DB2_PATH: "tasksTabledb2",
        DB3_PATH: "tasksTable"
    }[db_path]
    conn, c = get_conn(db_path)
    c.execute(f"SELECT DISTINCT name FROM {table}")
    return [row[0] for row in c.fetchall()]

def delete_by_name(db_path: str, name: str):
    table = {
        DB1_PATH: "tasksTabledb1",
        DB2_PATH: "tasksTabledb2",
        DB3_PATH: "tasksTable"
    }[db_path]
    conn, c = get_conn(db_path)
    c.execute(f"DELETE FROM {table} WHERE name = ?", (name,))
    conn.commit()

def find_by_name(db_path: str, name: str) -> Optional[Tuple]:
    table = {
        DB1_PATH: "tasksTabledb1",
        DB2_PATH: "tasksTabledb2",
        DB3_PATH: "tasksTable"
    }[db_path]
    conn, c = get_conn(db_path)
    c.execute(f"SELECT * FROM {table} WHERE name = ?", (name,))
    row = c.fetchone()
    return row

# ------------------------------
# Dataset generator (Haar cascade)
# ------------------------------
def generate_dataset1(name: str, id_: int):
    """Capture 120 grayscale cropped faces and save to data/<name>/."""
    if not os.path.exists(HAAR_PATH):
        st.error("Haar cascade not found in assets/. Add 'haarcascade_frontalface_default.xml'.")
        return
    os.makedirs(os.path.join("data", name), exist_ok=True)

    face_classifier = cv2.CascadeClassifier(HAAR_PATH)
    cap = cv2.VideoCapture(0)
    img_id = 0

    st.info("Starting camera—press ENTER to stop early.")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not available.")
            break

        gray = ensure_gray(frame)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            cv2.imshow("Frame", frame)
        else:
            for (x, y, w, h) in faces:
                cropped = frame[y:y+h, x:x+w]
                face = cv2.resize(cropped, (64, 64))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                img_id += 1
                file_path = os.path.join("data", name, f"{name}.{id_}.{img_id}.jpg")
                cv2.imwrite(file_path, face)
                cv2.putText(face, str(img_id), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Cropped face", face)

        if cv2.waitKey(1) == 13 or img_id >= 120:
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success("INSURANCE: Generating Dataset Completed.")

# ------------------------------
# UI Blocks
# ------------------------------
def sidebar_db_choice() -> Tuple[str, str]:
    """Return (db_label, db_path)."""
    menu3 = ["RSSB", "SORAS", "PRIME insurance"]
    choice3 = st.sidebar.selectbox("Please Choose a company DB:", menu3, index=0)
    db_map = {
        "SORAS": DB1_PATH,
        "PRIME insurance": DB2_PATH,
        "RSSB": DB3_PATH
    }
    return choice3, db_map[choice3]

def render_db_crud(company_label: str, db_path: str):
    st.title(f"INSURANCE: {company_label}")
    menu2 = ["Create", "Read", "Delete"]
    choice2 = st.sidebar.selectbox("Please Choose a Task:", menu2, index=0)

    if choice2 == "Create":
        st.subheader("Add a user/customer")
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120, step=1, value=25)
        with col2:
            address = st.text_input("Address")
            identity_numb = st.text_input("Identity Number")
        with col3:
            due_date = st.date_input("Due Date", value=datetime.today())
            uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
            preview_img = None
            file_bytes = None
            if uploaded:
                file_bytes = uploaded.getvalue()
                img_bgr = cv2_im_from_bytes(file_bytes)
                if img_bgr is not None:
                    img_rgb = ensure_rgb(img_bgr)
                    preview_img = Image.fromarray(img_rgb)

        if st.button("Need dataset"):
            if not name:
                st.warning("Please enter a name first.")
            else:
                # Auto-increment id
                rows = view_all_data(db_path)
                new_id = (max([r[0] for r in rows], default=0) + 1) if rows else 1
                os.makedirs(os.path.join("data", name), exist_ok=True)
                generate_dataset1(name, new_id)
                # store the uploaded photo as blob if provided
                photo_blob = file_bytes if uploaded else None
                add_data_named(db_path, new_id, name, address, age, identity_numb, str(due_date), photo_blob)
                st.success("Data Added!")

        if st.button("Just Register"):
            if not name:
                st.warning("Please enter a name.")
            else:
                rows = view_all_data(db_path)
                new_id = (max([r[0] for r in rows], default=0) + 1) if rows else 1
                photo_blob = file_bytes if uploaded else None
                add_data_named(db_path, new_id, name, address, age, identity_numb, str(due_date), photo_blob)
                st.success("Data Added!")

        if uploaded and preview_img is not None:
            st.write("Your Image")
            st.image(preview_img, use_column_width=True)

    elif choice2 == "Read":
        st.subheader("View customers")
        rows = view_all_data(db_path)
        df = pd.DataFrame(rows, columns=["id", "name", "address", "age", "identity_numb", "due_date", "photo"])
        st.dataframe(df, use_container_width=True)

    elif choice2 == "Delete":
        st.subheader("Delete a user by name")
        rows = view_all_data(db_path)
        df = pd.DataFrame(rows, columns=["id", "name", "address", "age", "identity_numb", "due_date", "photo"])
        with st.expander("View current data"):
            st.dataframe(df, use_container_width=True)
        names = view_unique_names(db_path)
        sel = st.selectbox("Name to delete", names)
        if st.button("Delete"):
            delete_by_name(db_path, sel)
            st.success(f"Deleted: {sel}")

def render_detection(filenames: List[str], features: List[np.ndarray], detector: MTCNN):
    prediction_mode = st.sidebar.radio("Mode", ("Single image", "Web camera"), index=0)

    if prediction_mode == "Single image":
        st.subheader("Who is this person?")
        uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        if uploaded:
            save_path = save_uploaded_image(uploaded)
            if not save_path:
                st.error("Failed to save the uploaded image.")
                return
            # Detect faces first
            img_bgr = cv2.imread(save_path)
            img_rgb = ensure_rgb(img_bgr)
            results = detector.detect_faces(img_rgb)
            if not results:
                st.warning("No human face detected.")
                st.image(Image.open(save_path), use_column_width=True)
                return

            # Extract embedding
            vec = extract_features(save_path, detector)
            if vec is None:
                st.warning("Could not extract features from this image.")
                return

            # Recommend
            idx, score = recommend(features, vec)
            matched_name = name_from_path(filenames[idx])
            similarity_pct = score * 100.0
            threshold = 55.0

            # Draw box and annotate
            (x, y, w, h) = results[0]["box"]
            draw_img = img_rgb.copy()
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

            if similarity_pct >= threshold:
                cv2.putText(draw_img, matched_name, (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Detected Face in uploaded image")
                    st.image(draw_img, use_column_width=True)
                with col2:
                    st.header(f"This face matches: {matched_name}")
                    st.image(filenames[idx], width=300)
                st.write(f"Similarity: **{similarity_pct:.2f}%**")

                # Lookup in DBs
                for label, db_path in [("SORAS", DB1_PATH), ("PRIME INSURANCE", DB2_PATH), ("RSSB", DB3_PATH)]:
                    row = find_by_name(db_path, matched_name)
                    if row:
                        st.subheader(f"Company: {label}")
                        id_, name, address, age, identity, due_date, photo = row
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.write("**ID**"); st.write(id_)
                            st.write("**NAME**"); st.write(name)
                            st.write("**ADDRESS**"); st.write(address)
                        with c2:
                            st.write("**AGE**"); st.write(age)
                            st.write("**ID CARD**"); st.write(identity)
                            st.write("**DATE**"); st.write(due_date)
                        with c3:
                            if photo:
                                file_byte = np.frombuffer(photo, dtype=np.uint8)
                                img = cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                                if img is not None:
                                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                        break
                else:
                    markdownstreamlit("Not Found in Any Databases", "h4")
            else:
                # Unknown
                cv2.putText(draw_img, "Unknown", (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                c3, c4 = st.columns(2)
                with c3:
                    st.header("Your uploaded image")
                    st.image(draw_img, use_column_width=True)
                with c4:
                    markdownstreamlit("UNKNOWN PERSON", "h3")

        else:
            st.info("Please upload an image.")

    else:  # Web camera mode
        st.subheader("Who is this person? (Web camera)")
        if st.button("Detect Face"):
            if not os.path.exists(HAAR_PATH):
                st.error("Haar cascade not found in assets/. Add 'haarcascade_frontalface_default.xml'.")
                return

            face_cascade = cv2.CascadeClassifier(HAAR_PATH)
            cap = cv2.VideoCapture(0)
            img_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera not available.")
                    break
                gray = ensure_gray(frame)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        detected = frame[y:y+h, x:x+w]
                        img_id += 1
                        file_path = os.path.join(FACES_DIR, f"user.{img_id}.jpg")
                        cv2.imwrite(file_path, detected)

                        # Build embedding from saved crop (or compute directly)
                        img_rgb = ensure_rgb(detected)
                        img_rgb = cv2.resize(img_rgb, (224, 224))
                        vec = extract_embedding_from_crop(img_rgb)
                        if vec is None:
                            continue

                        idx, score = recommend(features, vec)
                        matched_name = name_from_path(filenames[idx])
                        similarity_pct = score * 100.0
                        threshold = 55.0

                        draw = frame.copy()
                        color = (0, 255, 0) if similarity_pct >= threshold else (0, 0, 255)
                        label = matched_name if similarity_pct >= threshold else "Unknown"
                        cv2.rectangle(draw, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(draw, f"{label} ({similarity_pct:.1f}%)",
                                    (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        st.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB), caption="Live Detection", use_column_width=True)

                        # If matched, try to show DB info once
                        if similarity_pct >= threshold:
                            for label_db, db_path in [("SORAS", DB1_PATH), ("PRIME", DB2_PATH), ("RSSB", DB3_PATH)]:
                                row = find_by_name(db_path, matched_name)
                                if row:
                                    st.subheader(f"Company: {label_db}")
                                    id_, name, address, age, identity, due_date, photo = row
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        st.write("**ID**"); st.write(id_)
                                        st.write("**NAME**"); st.write(name)
                                        st.write("**ADDRESS**"); st.write(address)
                                    with c2:
                                        st.write("**AGE**"); st.write(age)
                                        st.write("**ID CARD**"); st.write(identity)
                                        st.write("**DATE**"); st.write(due_date)
                                    with c3:
                                        if photo:
                                            file_byte = np.frombuffer(photo, dtype=np.uint8)
                                            img = cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                                            if img is not None:
                                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                                    break

                # Exit quickly for demo (ENTER or 1 frame)
                if cv2.waitKey(1) == 13 or img_id >= 1:
                    break

            cap.release()
            cv2.destroyAllWindows()

# ------------------------------
# MAIN
# ------------------------------
def main():
    markdownstreamlit("Enhancing Customer Service Delivery in Insurance Companies", "h2")
    init_tables()  # ensure DBs/tables exist

    # Top menu
    menu1 = ["DETECTION", "DATABASE & DATASET", "FEATURES (Rebuild)"]
    choice1 = st.sidebar.selectbox("Please Choose a Section:", menu1, index=0)

    detector = get_detector()

    # Load filenames and features (or prompt to build)
    filenames = pickle.load(open(IMG_FILENAMES_PKL, "rb")) if os.path.exists(IMG_FILENAMES_PKL) else []
    features  = pickle.load(open(FEATURES_PKL, "rb")) if os.path.exists(FEATURES_PKL) else []

    if choice1 == "FEATURES (Rebuild)":
        st.info("Rebuild embeddings from images under data/<person_name>/")
        if st.button("Rebuild Now"):
            filenames, features = rebuild_features_from_data(detector)

        if filenames and features:
            st.success(f"Current index size: {len(filenames)} images.")
        else:
            st.warning("No index loaded. Go to 'Rebuild Now' or add pickles to artifacts/.")

    elif choice1 == "DATABASE & DATASET":
        company_label, db_path = sidebar_db_choice()
        render_db_crud(company_label, db_path)

    elif choice1 == "DETECTION":
        if not filenames or not features:
            st.warning("No features loaded. Go to 'FEATURES (Rebuild)' to build the index.")
        else:
            render_detection(filenames, features, detector)

if __name__ == "__main__":
    main()
