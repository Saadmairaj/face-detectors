# Face Detector

This repo contains various types of face detection techniques. All the face detection techniques are fine tunned and optimized out of the box to work the best with any resolution images and takes no time to get started

**Key features:**

- Easy to understand and setup
- Easy to manage
- Requires very less or no tuning for any resolution image
- No need to download models, they maintained automatically
- Uses ultralight face detection models that are very fast on CPU alone
- Get very good speed and accuracy on CPU alone
- All detectors share same parameters and methods, makes it easier to switch and go

**Detectors:**

- Hog detector
- CNN detector
- Caffemodel detector
- UltraLight 320 detector
- UltraLight 640 detector
  _( More on the way...)_

## Quick usage guide

Like said setup and usage is very simple and easy.

- Import the detector you want,
- Initialize it,
- Get predicts

**_Example_**

```python
from face_detector import Ultralight320Detector
from face_detector.utils import annotate_image

detector = Ultralight320Detector()

image = cv2.imread("image.png")

faces = detector.detect_faces(image)
image = annotate_image(image, faces, width=3)

cv2.imshow("view", image)
cv2.waitKey(100000)
```

## Performance guide

Every detector has different types of features

| Detector             | IMAGE 1 | IMAGE 2 | IMAGE 3 | IMAGE 4 | IMAGE 5 |
| -------------------- | ------- | ------- | ------- | ------- | ------- |
| Caffe Model          |         |         |         |         |         |
| CNN                  |         |         |         |         |         |
| Hog                  |         |         |         |         |         |
| UltraLight _(320px)_ |         |         |         |         |         |
| UltraLight _(640px)_ |         |         |         |         |         |
