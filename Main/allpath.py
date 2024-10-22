import os
model_path = os.path.join('.', 'Results', 'runs', 'detect', 'SGD_LR_0_005', 'weights', 'best.pt')
output_image_path = 'Detected_knife/captured_knife.jpg'
video_path = './video/video.mp4'
output_path = './video/output_with_boxes.mp4'