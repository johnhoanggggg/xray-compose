#!/usr/bin/env python3
"""Download the source images and normalise them to 8-bit grayscale PNGs in data/.

Most images come from the NCI Imaging Data Commons (public S3 bucket, all
CC BY 4.0), fetched with `idc-index`. The lower-leg film comes from
pydicom/pydicom-data on GitHub.

    pip install idc-index pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg numpy pillow
"""
import glob, io, os, tempfile, urllib.request
import numpy as np
import pydicom
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
P = "1.3.6.1.4.1.14519.5.2.1."

# name -> (SeriesInstanceUID, InstanceNumber or None, frame or None)
IDC = {
    "bonescan": (P + "1.29521004944781324998516417399786129744", None, 0),     # NM whole body, anterior
    "chest":    (P + "276957643109348739499848520765", None, None),             # CR chest PA
    "shoulder": (P + "293975334142452045390889095504828409586", None, None),    # DX shoulder AP
    "pelvis":   (P + "166567296538527715322688053521", None, None),             # CR pelvis AP
    "lspine":   (P + "289315498490798465477159944500", None, None),             # CR lumbar spine AP
    "elbow":    (P + "164612601394804586403360289159666509839", None, None),    # CR elbow AP + lateral
    "knee":     (P + "21941318645131582935505228063531101134", 1, None),        # CR knee AP
    "femur":    (P + "236953365119625415173352206913", 4, None),                # CR distal femur
    "head_mri": (P + "7009.2403.170855288003493623861643682590", 12, None),     # MR brain coronal T1
    "liver_us": (P + "1.17273761317913011857169890931336657148", 1536, None),   # US liver
}
PYDICOM_DATA = "https://raw.githubusercontent.com/pydicom/pydicom-data/master/data_store/data/"
GITHUB = {"leg": "RG3_UNCR.dcm"}                                               # CR tib/fib + ankle


def to_png(ds, frame, name):
    a = ds.pixel_array
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a.mean(-1)
    elif a.ndim == 3:
        a = a[frame or 0]
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.7)
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        a = 1 - a
    path = os.path.join(OUT, name + ".png")
    Image.fromarray((a * 255).astype(np.uint8)).save(path)
    print("wrote", path, a.shape)


def main():
    from idc_index import index
    os.makedirs(OUT, exist_ok=True)
    client = index.IDCClient()
    with tempfile.TemporaryDirectory() as tmp:
        uids = sorted({uid for uid, _, _ in IDC.values()})
        client.download_from_selection(seriesInstanceUID=uids, downloadDir=tmp,
                                       dirTemplate="%SeriesInstanceUID")
        for name, (uid, inst, frame) in IDC.items():
            for f in sorted(glob.glob(os.path.join(tmp, uid, "*.dcm"))):
                ds = pydicom.dcmread(f)
                if inst is None or int(ds.InstanceNumber) == inst:
                    to_png(ds, frame, name)
                    break
            else:
                print("missing", name)
    for name, fname in GITHUB.items():
        req = urllib.request.Request(PYDICOM_DATA + fname, headers={"User-Agent": "xray-compose"})
        with urllib.request.urlopen(req, timeout=120) as r:
            to_png(pydicom.dcmread(io.BytesIO(r.read())), None, name)


if __name__ == "__main__":
    main()
