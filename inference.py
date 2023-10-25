import cv2
from queue import Empty

from modules.config.config import config
from modules import Emotion
from modules.bbox.bbox import BoundingBox
from modules.detector import opencvwrapper
import cv2
import time
import numpy as np


def clearQueue(q):
    """clearQueue(q) > q
    @brief Delete buffer.
    @param q (cv::Mat): buffer.
    @returns q (cv::Mat): buffer.
    """
    try:
        while True:
            q.get_nowait()
    except Empty:
        pass


def serveStreaming(in_q):
    """serveStreaming(in_q)
    @brief Function to show frame in http when there's request via RESTful API.
    @param q (cv::Mat): buffer.
    @return frame (yield byte): formated-byte frame.
    """
    counter = 0
    while True:
        frame = in_q.get()
        # frame = cv2.resize(frame, (640, 360), interpolation = cv2.INTER_AREA) # arbitrary, for now
        counter += 1
        frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
        )
        in_q.task_done()


def serveProcessingAI(out_q, frame_queue):
    """serveProcessingAI(params, out_q, frame_queue)
    @brief Function to run AI apps without showing frame in http.
    @param params (string): variable to run stream or image.
    @param out_q (cv::Mat): buffer.
    @param frame_queue (int): number of frame in buffer.

    """
    while True:
        video = cv2.VideoCapture(config["source"])
        
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rsz_factor = config["resize_factor"]
        frame_width = int(frame_width * rsz_factor)
        frame_height = int(frame_height * rsz_factor)

        ret, frame = video.read()
        if ret: break

    track_id = 0
    last_bbox = None
    noface_count = 0

    max_fps = config["max_fps"]
    min_latency = 1 / max_fps

    prev_minute = time.localtime().tm_hour * 60 + time.localtime().tm_min

    count_frame = 0

    face_detector = opencvwrapper.build_model()

    emo_period = config["emo_period"]
    emo_tic = time.time()

    model_emo = Emotion.loadModel()
    # ============= MAIN: LOOP OVER VIDEO =============
    while True:
        toc = time.time()
        emo_update = False
        emo_update = (toc - emo_tic) >= emo_period
        if emo_update:
            emo_tic = toc
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)
        count_frame += 1
        if not ret:
            print("Not Connected to CCTV...")
            video = cv2.VideoCapture(config["source"])
        else:
            time_txt = ''
            frame = cv2.resize(frame, (frame_width, frame_height))

            start_time = time.time()
            local_time = time.localtime(start_time)

            start_minute = local_time.tm_hour * 60 + local_time.tm_min
            new_minute = False
            if start_minute != prev_minute:
                new_minute = True
                prev_minute = start_minute

            faces = opencvwrapper.detect_face(face_detector, frame)

            face_found = False
            for face in faces:
                face_img, face_bbox, conf = face
                if conf < 1:
                    continue
                face_found = True
                x, y, w, h = face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]
                break

            if face_found:
                noface_count = 0

                new_bbox = BoundingBox(x, y, w, h, track_id)
                
                if last_bbox is None:
                    track_id += 1
                    new_bbox.id = track_id
                elif new_bbox.distance(last_bbox) < 150:
                    new_bbox.c = last_bbox.c
                    new_bbox.name = last_bbox.name
                    new_bbox.emotion = last_bbox.emotion
                else:
                    track_id += 1
                    new_bbox.id = track_id
                last_bbox = new_bbox

                frame = new_bbox.draw(frame)
                w, h = cv2.getTextSize(str(track_id), 0, fontScale=0.8, thickness=2)[0]
                p1 = (new_bbox.x, new_bbox.y+new_bbox.h-4)
                p2 = p1[0] + w, p1[1] - h
                cv2.rectangle(frame, p1, p2, (13,17,17), -1, cv2.LINE_AA)
                cv2.putText(frame, 
                            str(track_id), 
                            (new_bbox.x, new_bbox.y+new_bbox.h-4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (255,255,255), 2,)     
            else:
                noface_count += 1
                if noface_count >= 10:
                    last_bbox = None


            if emo_update:
                if face_found:
                    # model_emo
                    img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    img_gray = cv2.resize(img_gray, (48, 48))
                    img_gray = np.expand_dims(img_gray, axis=0)
                    emotion_predictions = model_emo.predict(img_gray, verbose=0)[0, :]
                    dominant_emotion = Emotion.labels[np.argmax(emotion_predictions)]
                    new_bbox.emotion = dominant_emotion
            else:
                emo_time_left = int(emo_period - (toc - emo_tic) + 1)
                time_txt += f'{str(emo_time_left)}'

            if face_found:
                w, h = cv2.getTextSize(new_bbox.emotion, 0, fontScale=0.8, thickness=2)[0]
                p1 = (new_bbox.x, new_bbox.y + 20)
                p2 = p1[0] + w, p1[1] - h
                cv2.rectangle(frame, p1, p2, (57,0,139), -1, cv2.LINE_AA)
                cv2.putText(frame, 
                    new_bbox.emotion, 
                    (new_bbox.x, new_bbox.y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (229,238,246), 2,)
                
            cv2.putText(frame, 
                time_txt, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (229,238,246), 2,)
            

            # Count FPS
            end_time = time.time()  # End time for each frame
            elapsed_time = end_time - start_time
            if elapsed_time < min_latency:
                time.sleep(min_latency - elapsed_time)

            try:
                out_q.put(frame)
            except:
                raise -1

        if out_q.qsize() >= frame_queue:
            clearQueue(out_q)


if __name__ == "__main__":
    from queue import Queue
    q = Queue()
    serveProcessingAI(q, 3, {})
