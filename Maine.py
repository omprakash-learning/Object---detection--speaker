# STEP 1: Setup YOLOv5
!git clone https://github.com/ultralytics/yolov5  
%cd yolov5
!pip install -r requirements.txt
!pip install gTTS

# STEP 2: Import libraries
import torch
from IPython.display import Image, Audio, display
from google.colab import files
from gtts import gTTS
import shutil, os

# STEP 3: Upload an image
uploaded = files.upload()
for filename in uploaded.keys():
    shutil.move(filename, "test.jpg")

# STEP 4: Run object detection
!python detect.py \
  --weights yolov5s.pt \
  --source test.jpg \
  --imgsz 640 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --device cpu \
  --save-txt \
  --project runs/detect \
  --name exp \
  --exist-ok

# STEP 5: Class labels (COCO)
coco_classes = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard',
    'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',
    'scissors','teddy bear','hair drier','toothbrush'
]

# STEP 6: Extract and speak detected object names
labels_path = "runs/detect/exp/labels"
if os.path.exists(labels_path):
    label_files = os.listdir(labels_path)
    if label_files:
        label_file = os.path.join(labels_path, label_files[0])
        with open(label_file, 'r') as f:
            lines = f.readlines()

        detected_objects = []
        for line in lines:
            class_index = int(line.split()[0])
            detected_objects.append(coco_classes[class_index])

        if detected_objects:
            print("✅ Detected objects:", detected_objects)
            text = "Detected objects are: " + ", ".join(detected_objects)

            # Text-to-speech
            tts = gTTS(text)
            tts.save("detected.mp3")
            display(Audio("detected.mp3"))
        else:
            print("⚠️ No objects detected in labels.")
    else:
        print("⚠️ No label files found in:", labels_path)
else:
    print("❌ Labels folder does not exist.")
