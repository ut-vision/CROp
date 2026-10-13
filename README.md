# Constrained Rotation Optimization (CROp)

This repository provides a lightweight implementation of the preprocessing methods presented in our ECCV 2026 paper, Constrained Rotation Optimization: Revisiting Crop-Based Gaze Estimation.

The code is designed to be easy to integrate into existing appearance-based gaze-estimation pipelines. 
This repository focuses exclusively on preprocessing and does not include training or evaluation code.

It provides minimal, self-contained implementations of:

- `crop.py`: CROp preprocessing, the crop-based approach introduced in this work.
- `normalization.py`: standard face normalization, included as a baseline for comparison.

The only runtime dependencies are NumPy, OpenCV, and SciPy:
```bash
pip install numpy opencv-python scipy
```

## Usage

The package exposes both preprocessing methods:

```python
from crop import crop_preprocess
from normalization import normalize_preprocess
```


Both methods return an object with the same fields, allowing downstream code to switch between preprocessing strategies with minimal interface changes:

- `image`: preprocessed face patch
- `rotation`: rotation from original camera coordinates to patch coordinates
- `transform_matrix`: image-space homography from original pixels to patch pixels
- `headpose_matrix`: transformed head-pose rotation, if provided
- `gaze_vector`: transformed 3D gaze vector, if provided
- `gaze_pitchyaw`: transformed pitch/yaw gaze label, if provided
- `landmarks`: transformed landmarks, if provided

```python
result_crop = crop_preprocess(
    image=image,
    camera_matrix=camera_matrix,
    bbox=bbox,
    gaze_vector=gaze_vector,  # optional label to transform
)

result_normalization = normalize_preprocess(
    image=image,
    camera_matrix=camera_matrix,
    face_center=face_center,
    headpose=headpose,
    landmarks=landmarks,
    gaze_vector=gaze_vector,  # optional label to transform
)
```

## CROp vs. Normalization

`crop_preprocess` requires only an image, camera intrinsics, and a face bounding
box. The bounding box can be specified as `(x1, y1, x2, y2)` or as four 2D corner points with shape
`(4, 2)`. Bounding-box extraction is intentionally left to the caller, so the implementation can be used with any face detector or annotation
format.

Head pose, gaze, and landmarks are optional labels. When provided, head pose and
gaze are transformed into the CROp coordinate system, while 2D landmarks are translated and
resized into crop-pixel coordinates. Gaze is expected as 
a 3D direction vector.

`normalize_preprocess` implements the classic virtual-camera warp introduced in [*Revisiting Data Normalization for Appearance-Based Gaze Estimation*](https://dl.acm.org/doi/10.1145/3204493.3204548) by Zhang, Sugano, and Bulling (ETRA 2018). 
It is included as a baseline for comparison and reproducibility; it is not a contribution of this work. 
Unlike CROp, classic normalization requires a 3D face center and head pose to define the virtual camera.


## Citation

If you find this code useful, please consider citing our paper:
```bibtex
@inproceedings{santambrogio2026crop,
  title     = {Constrained Rotation Optimization: Revisiting Crop-Based Gaze Estimation},
  author    = {Santambrogio, Riccardo and Qin, Jiawei and Matteucci, Matteo and Sugano, Yusuke},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
  pages     = {362--379},
  publisher = {Springer},
  doi       = {10.1007/978-3-032-37232-1_20}
}
```
