import os
model_path = os.path.join('.', 'Results', 'runs', 'detect', 'ADAMW_LR_0_001', 'weights', 'last.pt')
output_image_path = 'Detected_knife/captured_knife.jpg'
video_path = './video_and_photo/video.mp4'
image_path = './video_and_photo/photo/2.jpg'
output_path = './video_and_photo/output_with_boxes.mp4'