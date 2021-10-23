import cv2


def annotate_image(image, faces, box_rgb=(100, 0, 255), keypoints_rgb=(255, 100, 0), width=2):
    """Annotate image with the payload (faces detection)

    Args:
        image (numpy.Array): Give image for annotation
        faces (list): Payload returned by detector.detect_faces or 
            detector.detect_faces_keypoints
        box_rgb (tuple, optional): RGB color for rectangle to be of. 
            Defaults to (100, 0, 255).
        keypoints_rgb (tuple, optional): RGB color for keypoints to be of. 
            Defaults to (150, 0, 255).
        width (int, optional): Width of annotations. Defaults to 2.

    Returns:
        numpy.Array: Image with annotations
    """

    img_height, img_width = image.shape[:2]
    ratio = ((img_height / 1080) + (img_width / 1920)) / 2
    width = max(1, round(width * ratio))

    for face in faces:
        if 'bbox' in face:
            (top, right, bottom, left) = face['bbox']
            image = cv2.rectangle(
                image, (left, top), (right, bottom), box_rgb, thickness=width)

        if 'keypoints' in face:
            for keypoint in face['keypoints'].values():
                image = cv2.circle(
                    image, keypoint, width, keypoints_rgb, -1)
    return image
