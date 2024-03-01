import numpy as np
from PIL import Image
import cv2
import tensorflow as tf


def segmentation(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128))
    frame = np.array(frame)

    prediction = model.predict(np.expand_dims(frame, 0))
    predicted_img = np.argmax(prediction, axis=3)[0,:,:]
    predicted_img = predicted_img.astype(np.uint8)

    segmented_frame = np.zeros_like(frame)
    segmented_frame[:, :, 0] = predicted_img * 255

    segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR)

    return segmented_frame

def make_heatmap(frame):
    heatmap = np.zeros_like(frame[..., 0], dtype=np.float32)

    for x in range(frame.shape[0]):
        for y in range(frame.shape[1]):
            if frame[:,:,2][x][y]>0:
                heatmap[x, y] += frame[:,:,2][x][y]
            
                if heatmap[x, y] > 255:
                    heatmap[x, y] = 255

            elif x>(frame.shape[0]-frame.shape[0]/2.7):
                heatmap[x, y] += 100
            
                if heatmap[x, y] > 255:
                    heatmap[x, y] = 255
            
            else:
                heatmap[x, y] -= 5

                if heatmap[x, y] < 0:
                    heatmap[x, y] = 0
    
    heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_RAINBOW)

    return heatmap_colored

def motion_zone_points_maker(mask):
    stretched_mask = cv2.resize(mask, (848, 480))
    gray_mask = np.zeros_like(stretched_mask[:,:,0])
    gray_mask = stretched_mask[:,:,2]

    obstacle_indices = np.where(gray_mask > 200)

    motion_zone_points = []

    for y in range(int(480 * 0.37), int(480 * 0.63)):
        x_values = obstacle_indices[1][obstacle_indices[0] == y]
        right_half_x_values = x_values[x_values > 424]

        if len(right_half_x_values) > 0:
            min_x = np.min(right_half_x_values)

            motion_zone_points.append((min_x, y))
        else:
            motion_zone_points.append((424, y))
    
    return motion_zone_points

def create_motion_zone(mask, frame):
    motion_zone_points = motion_zone_points_maker(mask)

    cv2.line(frame, motion_zone_points[0], (424, int(480 * 0.37)), (0, 255, 0), 2)
    cv2.line(frame, motion_zone_points[-1], (424, int(480 * 0.63)), (0, 255, 0), 2)

    for point in motion_zone_points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
    
    for i in range(len(motion_zone_points) - 1):
        cv2.line(frame, motion_zone_points[i], motion_zone_points[i + 1], (255, 0, 0), 2)
    
    return frame


model = tf.keras.models.load_model('./models/resnet_backbone.hdf5')
video_capture = cv2.VideoCapture('./input_video.mp4')

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    segmented_frame = segmentation(frame)

    heatmap = make_heatmap(segmented_frame)

    motion_zone_frame = create_motion_zone(heatmap, frame)

    cv2.imshow('Motion Zone', motion_zone_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


