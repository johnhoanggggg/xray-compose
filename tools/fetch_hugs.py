#!/usr/bin/env python3
"""Download the embrace photos and the MediaPipe models that tools/veins.py needs.

The photos are from the Open Images V7 validation set: images that human
raters tagged "Hug" (/m/025s9qt). All of them are Flickr photos licensed
CC BY 2.0. The models are Google's MediaPipe pose landmarker, BlazeFace face
detector and multiclass selfie segmenter.

    pip install mediapipe opencv-python-headless numpy scipy
"""
import os, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OPEN_IMAGES = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"
MODELS = {
    "pose_landmarker_heavy.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "selfie_multiclass_256x256.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}

# Open Images id -> (Flickr author, title, Flickr page). All CC BY 2.0.
HUGS = {
    "1ca2486498488b59": ("SharonaGott", "sharona and david", "https://www.flickr.com/photos/gottshar/14429153530/"),
    "e881775b897cafa8": ("Greg Hernandez", "Darryl Stephens and Peter Paige", "https://www.flickr.com/photos/greginhollywood/9336014111"),
    "dd5669c3169e107a": ("mat's eye", "bf friendship", "https://www.flickr.com/photos/matte0ne/5030087258/"),
    "f14f17cd97f06d27": ("llinddsayy", "hug hug kiss kiss", "https://www.flickr.com/photos/ohsugarplum/7996768358"),
    "1820959df8c1fe35": ("monicasecas", "Untitled", "https://www.flickr.com/photos/monicasecas/5826338411"),
    "5807ea2ff3607943": ("Patty", "Picture_343", "https://www.flickr.com/photos/pattista/222358755"),
    "949142002afa77e1": ("LeAnn E. Crowe", "Brian and Me", "https://www.flickr.com/photos/technicolor76/4326888954"),
    "bc1dfc4050f7c681": ("Gabriela Pinto", "Mom & Son", "https://www.flickr.com/photos/gabrielap93/6957898246"),
}


def get(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(url) as r, open(path + ".part", "wb") as f:
        f.write(r.read())
    os.replace(path + ".part", path)
    print("wrote", path)


def main():
    for name, url in MODELS.items():
        get(url, os.path.join(ROOT, "models", name))
    for iid in HUGS:
        get(OPEN_IMAGES.format(iid), os.path.join(ROOT, "hugs", iid + ".jpg"))


if __name__ == "__main__":
    main()
