import dlib
from face_detectors.detectors.utils import BaseModel, _rect_to_css, FaceDetectorModels


class CNNDetector(BaseModel):
    """
    CNN detector is an high level client for 
    dlib::cnn_face_detection_model_v1 that is simpler 
    and easier to manage
    """

    NAME = "CNN detector"
    model_path = None

    def __init__(self, convert_color=None,
                 number_of_times_to_upsample=1, confidence=0.5, **kw):
        """Convolutional Neural Network (CNN) detector constructor. 
        This detector will not behave much different if compared to 
        dlib::cnn_face_detection_model_v1.

        Args:
            convert_color (int, optional): Takes OpenCV COLOR codes to convert the images. 
                Defaults to cv2.COLOR_BGR2RGB.
            number_of_times_to_upsample (int, optional): Up samples the image 
                number_of_times_to_upsample before running the basic detector. By default is 1.
            confidence (float, optional): Confidence score is used to refrain from making 
                predictions when it is not above a sufficient threshold. Defaults to 0.5.
            scale (float, optional): Scales the image for faster output (No need to set 
                this manually, scale will be determined automatically if no value is given)
        """
        self.scale = kw.get('scale', 0.25)
        self.confidence = confidence
        self.convert_color = convert_color
        self.number_of_times_to_upsample = number_of_times_to_upsample

        self._setup(**kw)

    def _setup(self, **kw):
        """Internal function, do not call directly.

        Setup the model with the configurations"""
        self.model_path = FaceDetectorModels.absolute_path(
            FaceDetectorModels.cnn_mmod_face_detector)

        assert self.scale is not None, "Scale is required"
        assert self.model_path is not None, "Model path is empty"

        self.detector = dlib.cnn_face_detection_model_v1(self.model_path)
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
        preprocessed_faces = self.detector(
            image, self.number_of_times_to_upsample)
        faces = [{"bbox": _rect_to_css(face.rect), "confidence": face.confidence}
                 for face in preprocessed_faces if face.confidence > self.confidence]
        return self._scale_back(faces=faces, scale=scale)
