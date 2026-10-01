#!/usr/bin/env python3
"""Download public medical images and normalise them to 8-bit grayscale PNGs.

Sources (all reachable via raw.githubusercontent.com):
  * X-ray : ieee8023/covid-chestxray-dataset (PA / AP / lateral chest films)
  * CR    : pydicom/pydicom-data RG1 (chest CR) + RG3 (extremity CR)
  * MRI   : pydicom/pydicom-data MR2_UNCR (brain) + emri_small (10 head slices)
  * US    : pydicom/pydicom-data US1_UNCR (ultrasound frame)

Output: data/<modality>_<n>.png  (modality in xray, mri, us)
"""
import csv, io, os, sys, urllib.request
import numpy as np
from PIL import Image
import pydicom

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
CXR = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/"
DCM = "https://raw.githubusercontent.com/pydicom/pydicom-data/master/data_store/data/"
N_XRAY = int(os.environ.get("N_XRAY", "12"))

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "xray-compose"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

def norm(a, invert=False):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if invert:
        a = 1 - a
    return (a * 255).astype(np.uint8)

counts = {}
def save(mod, arr):
    n = counts.get(mod, 0); counts[mod] = n + 1
    p = os.path.join(OUT, f"{mod}_{n:02d}.png")
    Image.fromarray(arr).save(p)
    print("wrote", p, arr.shape)

def dicom(name, mod, frames=None):
    d = pydicom.dcmread(io.BytesIO(get(DCM + name)))
    a = d.pixel_array
    inv = d.PhotometricInterpretation == "MONOCHROME1"
    if a.ndim == 3 and a.shape[-1] == 3:          # RGB ultrasound
        a = a.astype(np.float32).mean(-1)
        save(mod, norm(a)); return
    if a.ndim == 3:                                # multi-frame
        for i in (frames if frames is not None else range(a.shape[0])):
            save(mod, norm(a[i], inv))
        return
    save(mod, norm(a, inv))

def main():
    os.makedirs(OUT, exist_ok=True)
    dicom("MR2_UNCR.dcm", "mri")
    dicom("emri_small.dcm", "mri", frames=[2, 4, 6, 8])
    dicom("US1_UNCR.dcm", "us")
    dicom("RG1_UNCR.dcm", "xray")
    dicom("RG3_UNCR.dcm", "xray")

    rows = list(csv.DictReader(io.StringIO(get(CXR + "metadata.csv").decode())))
    rows = [r for r in rows if r["modality"] == "X-ray" and r["folder"] == "images"
            and r["filename"].lower().endswith((".jpg", ".jpeg", ".png"))]
    picked, views = [], {}
    for r in rows:                                  # round-robin over views
        v = r["view"]
        if views.get(v, 0) < N_XRAY // 3 + 1:
            views[v] = views.get(v, 0) + 1; picked.append(r)
        if len(picked) >= N_XRAY: break
    for r in picked:
        try:
            im = Image.open(io.BytesIO(get(CXR + "images/" + r["filename"]))).convert("L")
            im.thumbnail((1024, 1024))
            save("xray", norm(np.asarray(im)))
        except Exception as e:
            print("skip", r["filename"], e, file=sys.stderr)

if __name__ == "__main__":
    main()
