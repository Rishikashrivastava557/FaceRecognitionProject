import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

# Constants
TRAINING_DIR = "trainingImages"
EIGENFACES_DIR = "eigenfaces"
FACE_WIDTH, FACE_HEIGHT = 125, 150
NUM_EIGENFACES = 20
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"

# Ensure directories exist
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(EIGENFACES_DIR, exist_ok=True)

# Face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Utility functions
def resize_gray(image, width=FACE_WIDTH, height=FACE_HEIGHT):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height))
    return resized

def load_training_images():
    images = []
    filenames = []
    for fn in sorted(os.listdir(TRAINING_DIR)):
        if fn.endswith(".png"):
            path = os.path.join(TRAINING_DIR, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (FACE_WIDTH, FACE_HEIGHT))
                images.append(img_resized.flatten())
                filenames.append(fn)
    return np.array(images), filenames

# Build Eigenfaces
def build_eigenfaces():
    images, filenames = load_training_images()
    if len(images) < 2:
        messagebox.showerror("Error", "Need at least 2 training images")
        return None, None, None

    pca = PCA(n_components=NUM_EIGENFACES, whiten=True)
    pca.fit(images)

    # Save eigenfaces as images
    for i, eigenface in enumerate(pca.components_):
        ef_img = eigenface.reshape(FACE_HEIGHT, FACE_WIDTH)
        ef_img = cv2.normalize(ef_img, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(EIGENFACES_DIR, f"eigen_{i}.png"), ef_img)

    messagebox.showinfo("Info", f"Built {NUM_EIGENFACES} eigenfaces")
    return pca, images, filenames

# Recognize face
def recognize_face(pca, images, filenames, face_img):
    if pca is None:
        messagebox.showerror("Error", "Eigenfaces not built yet")
        return None, None

    face_vec = face_img.flatten().reshape(1, -1)
    face_proj = pca.transform(face_vec)

    # Project training images
    train_proj = pca.transform(images)

    # Compute Euclidean distances
    dists = np.linalg.norm(train_proj - face_proj, axis=1)
    min_idx = np.argmin(dists)
    min_dist = dists[min_idx]

    # Threshold can be tuned
    threshold = 3000
    if min_dist < threshold:
        return filenames[min_idx], min_dist
    else:
        return None, None

# Face capture and save
def capture_faces():
    cap = cv2.VideoCapture(0)
    count = 0
    messagebox.showinfo("Info", "Press 's' to save face, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Capture Faces", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if len(faces) == 0:
                print("No face detected")
                continue
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]
            face_img = resize_gray(face_img)
            filename = os.path.join(TRAINING_DIR, f"face_{count}.png")
            cv2.imwrite(filename, face_img)
            print(f"Saved {filename}")
            count += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Face recognition from camera
def recognize_from_camera(pca, images, filenames):
    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Info", "Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = resize_gray(face_img)
            name, dist = recognize_face(pca, images, filenames, face_img)
            label = name if name else "Unknown"
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI
class MainApp:
    def __init__(self, root):
        self.root = root
        root.title("Face Recognition")

        self.pca = None
        self.images = None
        self.filenames = None

        self.btn_capture = Button(root, text="Take Picture", command=capture_faces)
        self.btn_capture.pack(pady=10)

        self.btn_build = Button(root, text="Build Eigenfaces", command=self.build_eigenfaces)
        self.btn_build.pack(pady=10)

        self.btn_recognize = Button(root, text="Recognize", command=self.recognize)
        self.btn_recognize.pack(pady=10)

    def build_eigenfaces(self):
        self.pca, self.images, self.filenames = build_eigenfaces()

    def recognize(self):
        if self.pca is None:
            messagebox.showerror("Error", "Build eigenfaces first")
            return
        recognize_from_camera(self.pca, self.images, self.filenames)

if __name__ == "__main__":
    root = Tk()
    app = MainApp(root)
    root.mainloop()python face_recognition_app.py
