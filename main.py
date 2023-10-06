import helper_functions as hf
import pandas as pd

# set params
chosen_model = "yolo-v5.tflite"
chosen_source_directory = 'test_videos_to_process'
chosen_output_directory = 'processed_videos'
chosen_frame_breaks = 600 # roughly 10s of video
d = {'class': [15, 77],'class_name': ['cat','teddy bear'] , 'threshold': [0.5, 0.5]} # see list here: https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
my_criteria = pd.DataFrame(data=d)

# execute
hf.process_directory(chosen_model = chosen_model,
	chosen_source_directory = chosen_source_directory,
	chosen_output_directory = chosen_output_directory,
	chosen_frame_breaks = chosen_frame_breaks,
	my_criteria = my_criteria)


