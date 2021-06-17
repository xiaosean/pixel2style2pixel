import time
import glob

import os
import cv2

from DSFD import face_detection


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

def crop_image(im, bboxes):
    faces = []
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        faces += [im[y0:y1, x0:x1]]
    return faces



if __name__ == "__main__":
    # impaths = "images"
    # outpaths = "./out/"
    # impaths = glob.glob(os.path.join(impaths, "*.png"))
    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=1080
    )
    # VIDEO = "raw_data/Timeline 1_10m_36_rgb.mov"
    # VIDEO = "raw_data/ir_0312.mp4"
    VIDEO = "./00080_.mp4"
    OUT_DIR = "./out"
    cap = cv2.VideoCapture(VIDEO)
    cnt = 0
    t = time.time()
    while(cap.isOpened()):
        cnt += 1
        ret1, im = cap.read()   
        dets = detector.detect(
            im[:, :, ::-1]
        )[:, :4]
        if cnt < 1000:
            continue

        if cnt % 100 ==0 :
            print(f"elapse time: {time.time()- t:.3f}")

        if cnt % 30 ==0 :
            faces = crop_image(im, dets)
            for idx, face in enumerate(faces):
                output_path = os.path.join(
                    OUT_DIR,
                    f"{cnt}_{idx}_out.png"
                )

                cv2.imwrite(output_path, face)
