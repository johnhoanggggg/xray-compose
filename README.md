# xray-compose

Splices X-ray, MRI and ultrasound images into a blocky human silhouette, keeping the hard rectangular tile borders.

```sh
pip install pydicom numpy pillow
make data   # download source images into data/
make run    # build the C++ compositor and write out/composite.png
```

- `tools/fetch.py` pulls chest X-rays from [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset), plus CR/MRI/ultrasound DICOMs from [pydicom/pydicom-data](https://github.com/pydicom/pydicom-data), and normalises them to 8-bit PNGs.
- `src/compose.cpp` (C++17, uses the stb headers in `third_party/`) draws a soft speckled body outline and pastes each tile from `layout.txt` as a hard-edged rectangle.
- `layout.txt` has one tile per line: `file x y w h [sx sy sw sh] [rot90] [border]`. Positions are fractions of the canvas, and the optional source crop is a fraction of the image.

Downloaded images keep their original licenses, so `data/` and `out/` are git-ignored.
