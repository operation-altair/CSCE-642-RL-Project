from ultralytics import YOLO
import cv2
import torch
# load YOLOv8 model
def predict_image (model_path,image_path):   #thia is for image
    list = []
    model = YOLO(model_path)
    # read the img
    img = cv2.imread(image_path)
    # predict
    results = model(img)
    # print result
    xywh = results[0].boxes.xywh.cpu()
    cls = results[0].boxes.cls.cpu()
    combined = torch.cat((xywh, cls.unsqueeze(1)), dim=1)   #combined is a tensor, it has 5 column, is the xy center of the box,width, high,and class(1 is hen,0 is egg),

    for i in range(combined.size(0)):
        grid_x = int((combined[i, 0] / 848)*16)
        grid_y = int((combined[i, 1] / 480)*16)
        classes = int(combined[i, -1])
        list.append({"grid_x": grid_x, "grid_y": grid_y, "class": classes})
    print (list)
    return list

def predict_video (model_path,video_path):  #this is for video
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        xywh = results[0].boxes.xywh.cpu()
        cls = results[0].boxes.cls.cpu()
        combined = torch.cat((xywh, cls.unsqueeze(1)), dim=1)  #combined is a tensor, it has 5 column, is the xy center of the box,width, high,and class(1 is hen,0 is egg),
        print(combined)

# predict_image(r'D:\Python_Project\chicken_detected\runs\pose\train30\weights\best.pt',
#               r"D:\Python_Project\chicken_detected\data\depth\early_fusion\RGBD\images\train\20240226_082700_png.rf.b30afe849dd28d3eea6bfb61c7ba229f_rb.jpg"
#               )