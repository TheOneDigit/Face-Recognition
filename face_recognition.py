from deepface import DeepFace
import numpy as np
import cv2
from tqdm.notebook import tqdm



def face_recognition(video_path):

  backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
  models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
  metrics = ["cosine", "euclidean", "euclidean_l2"]

  vid = cv2.VideoCapture(video_path)
  frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('processed_video.avi', fourcc, 20.0, (frame_width, frame_height))
  total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
  pbar = tqdm(total=total_frames)


  while vid.isOpened():
    ret, frame = vid.read()
      
    if not ret:
      break
    try:
      people = DeepFace.find(img_path=frame, db_path="./my_db", model_name=models[2],
                        distance_metric=metrics[1], enforce_detection=True, silent=True, detector_backend='yolov8')
      for person in people:
        select_face = person.loc[person['distance'].idxmax()]
        x,y,w,h = select_face['source_x'], select_face['source_y'], select_face['source_w'], select_face['source_h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        name = os.path.basename(select_face['identity']).split('.')[0][:-1].capitalize()
        cv2.putText(frame, name, (x, y), cv2.FONT_ITALIC, 2, (0, 0, 255), 3)
    except Exception as e:
      print("The error is ",e)
      pass
    out.write(frame)
    pbar.update(1)

  vid.release()
  out.release()
  cv2.destroyAllWindows()

  return 'Video Processed successfully'
