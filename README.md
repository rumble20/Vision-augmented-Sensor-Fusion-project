# Vision-augmented-Sensor-Fusion-project

Small python pipeline that fuses a deep learning object detector with an
extended kalman filter, so a target keeps being tracked (and its path
predicted) when the camera drops frames.

## how it works

- `tracker/detector.py`: detector interface. `YoloDetector` wraps an
  ultralytics yolo model and returns the bbox centre of the best box.
  `SimulatedDetector` gives noisy centres and nothing on dropped frames.
- `tracker/ekf.py`: ekf with a constant turn rate and velocity model,
  state `[x, y, v, yaw, yaw_rate]`, measurement is the bbox centre.
- `tracker/pipeline.py`: predict on every frame, update only if there is a
  detection and it passes a chi2 gate. On a dropped frame the filter just
  coasts on the motion model. `predict_path(n)` rolls it forward for the
  trajectory prediction.

## run

```
pip install -r requirements.txt
python simulate.py
python -m pytest
```

`simulate.py` drives a target along an arc, blacks the camera out for 20
frames (2 s) and compares against holding the last detection.

## results (simulated, 100 seeds, 3 px detection noise)

rmse over the 20 dropped frames:

| method               | median | worst   |
|----------------------|--------|---------|
| ekf coasting         | 11 px  | 17 px   |
| hold last detection  | 48 px  | 55 px   |

## not done / not tested

- The yolo wrapper needs `pip install ultralytics` and weights, it is not
  in requirements and it has **not been run**, only the simulated detector is
  tested. Nothing was run on real video.
- Single target, 2d pixel coordinates, no camera model or data association.
- The noise parameters are hand picked, not tuned.
