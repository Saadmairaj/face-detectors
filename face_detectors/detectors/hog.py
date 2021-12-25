import dlib
from face_detectors.detectors.utils import BaseModel, _rect_to_css


class HogDetector(BaseModel):
    """
    Hog detector is an hight level client for 
    dlib::get_frontal_face_detector that is fine tuned 
    by default which is more accurate, optimized and 
    modular out of the box.
    """

    NAME = "Hog Detector"

    def __init__(self, convert_color=None, number_of_times_to_upsample=2, confidence=0.5, **kw):
        """Hog detector constructor. 

        Args:
            convert_color (int, optional): Takes OpenCV COLOR codes to convert the images. 
                Defaults to cv2.COLOR_BGR2RGB.
            number_of_times_to_upsample (int, optional): Upsamples the image 
                number_of_times_to_upsample before running the basic detector. By default is 2.
            confidence (float, optional): Confidence score is used to refrain from making 
                predictions when it is not above a sufficient threshold. Defaults to 0.5.
            scale (float, optional): Scales the image for faster output (No need to set 
                this manually, scale will be determined automatically if no value is given)
        """
        self.scale = kw.get('scale', "0.25D")
        self.confidence = confidence
        self.convert_color = convert_color
        self.number_of_times_to_upsample = number_of_times_to_upsample

        self._setup(**kw)

    def _setup(self, **kw):
        """Internal function, do not call directly.

        Setup the model with the configurations"""
        assert self.scale != None, "Scale is required"

        self.detector = dlib.get_frontal_face_detector()
        self._setup = True
        return self

    def detect_faces(self, image):
        """Detects faces in the given image

        Args:
            image: Give numpy.array image
        Return:
            List[Any, List]: List of faces coordinates"""
        assert getattr(self, '_setup', None) != None, "The model is not setup."

        h, w = image.shape[:2]

        if isinstance(self.scale, str) and self.scale[-1] == "D":
            scale = float(self.scale[:-1])
            if (w > h and w < 650) or (h > w and h < 650):
                scale = 1
        else:
            scale = self.scale
        image = self._prep_image(image, scale=scale)
        face_rects, scores, _ = self.detector.run(
            image, self.number_of_times_to_upsample, -0.35)
        face_rects = [_rect_to_css(face) for face in face_rects]

        faces = []
        for face, score in zip(face_rects, scores):
            if score < 0:
                score = 1 + score
            score = min(1, score * 5)  # score correction
            if score > self.confidence:
                faces.append(
                    {'bbox': list(face), 'confidence': score}
                )
        return self._scale_back(faces=faces, scale=scale)
