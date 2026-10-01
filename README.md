# xray-compose

Splices real X-ray, MRI and ultrasound images over a whole-body nuclear bone scan so each tile sits on its matching anatomy. Every tile keeps a hard rectangular border.

![composite](out/composite.png)

```sh
pip install idc-index pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg numpy pillow
make data   # download the source DICOMs and write data/*.png
make run    # build the C++ compositor and write out/composite.png
```

## How it works

- `tools/fetch.py` pulls specific series from the [NCI Imaging Data Commons](https://imaging.datacommons.cancer.gov/) through `idc-index`:
  - a whole-body bone scan
  - CR/DX films of the chest, shoulder, pelvis, lumbar spine, elbow, knee and femur
  - a coronal brain MRI
  - a liver ultrasound

  It also pulls a lower-leg CR film from [pydicom-data](https://github.com/pydicom/pydicom-data). Each image is normalised to an 8-bit PNG.
- `src/compose.cpp` (C++17, with the stb headers in `third_party/`) draws the inverted bone scan as the body. It then pins each tile by landmark: a point in the source image, such as the midpoint between the femoral heads, maps onto the same anatomy on the bone scan, with a given width and rotation.
- `layout.txt` sets the background and the tiles:

  ```
  bg   file invert gain gamma x y
  tile file  sx sy sw sh  ax ay  dx dy  width angle [border]
  ```

  - `s*`: crop, as fractions of the source image.
  - `a*`: source landmark, as fractions of the source image.
  - `d*` and `width`: target position and tile width, as percentages of the 512×1088 bone-scan space.
  - Tiles are drawn in file order.

## Credits

The IDC images come from TCIA collections CMB-PCA, CMB-MML, CMB-LCA and VAREPOP-APOLLO, licensed CC BY 4.0. The leg film is from pydicom-data.
