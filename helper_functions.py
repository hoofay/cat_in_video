import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import glob
from pathlib import Path
import time
import os

# load the model
def load_model(model_path):

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return input_details, output_details, interpreter

# get videos from a directory
def get_videos(directory):
    directory = 'test_videos_to_process'
    video_files = glob.glob(os.path.join(directory, '*.mp4'))
    return video_files

# set some names
def set_output_name(video_to_process,output_directory):
    name_prefix = Path(video_to_process).stem
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_video_file = output_directory + '//' + name_prefix + "_out_" + str(timestr) + ".mp4"
    
    return output_video_file

# open connections to cv2
def open_video_streams(video_to_process,output_file_name):
    # Open the video stream
    video_capture = cv2.VideoCapture(video_to_process)
    
    # Define the output video file name and settings
    output_frame_size = (int(video_capture.get(3)), int(video_capture.get(4)))  # Use the same size as the input video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_file_name, fourcc, 30.0, output_frame_size)

    print('video connections opened')
    return video_capture, output_video

# define model output processing functions

# class filter
def class_filter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

# yolo detect
def yolo_detect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                
    boxes = np.squeeze(output_data[..., :4])    
    scores = np.squeeze( output_data[..., 4:5]) 
    classes = class_filter(output_data[..., 5:]) 
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] # xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]


# define running the model and outputs
def process_video_with_ml(video_capture, interpreter, input_details, output_details, output_video, max_frame_break=50):

    # initialise an empty list to store results
    results = []

    # initialise a frame counter
    frame_count = 0
    
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Store the original frame dimensions for later conversion
        original_frame_height, original_frame_width, _ = frame.shape

        # Preprocess the frame (resize, normalize, etc.) to match the model's input shape
        resized_frame = cv2.resize(frame,(320,320))
        normalized_frame = resized_frame / 255.0
        preprocessed_frame = np.expand_dims(normalized_frame, axis=0).astype(np.float32)

        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
        interpreter.invoke()
        detections = interpreter.get_tensor(output_details[0]['index'])
        xyxy, classes, scores = yolo_detect(detections) #boxes(x,y,x,y), classes(int), scores(float)

        # draw on video and write results 
        for i in range(len(scores)):
            if ((scores[i] > 0.3) and (scores[i] <= 1.0)):
                H = frame.shape[0]
                W = frame.shape[1]
                xmin = int(max(1,(xyxy[0][i] * W)))
                ymin = int(max(1,(xyxy[1][i] * H)))
                xmax = int(min(H,(xyxy[2][i] * W)))
                ymax = int(min(W,(xyxy[3][i] * H)))

                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                cv2.putText(frame, f"{classes[i]}: {scores[i]:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                results.append({"Frame":frame_count,"Class": classes[i], "Confidence": scores[i]})

        # Write the processed frame to the output video
        output_video.write(frame)

        # Increment the frame counter
        frame_count += 1

        if frame_count % 10 == 0:
            print(frame_count)

        # Check if the maximum frame count has been reached
        if frame_count >= max_frame_break:
            break
        
    print('video processing complete')
    results_df = pd.DataFrame(results)
    return results_df

def close_video_connections(video_capture,output_video):
    # close video connections
    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()
    print('connections closed')

def is_in_video(class_id,threshold,results):
    checks = []
    for i in results.index:
        if results.iloc[i,1] == class_id and results.iloc[i,2] >= threshold:
            checks.append(True)
        else:
            checks.append(False)
    if True in checks:
        return True
    else:
        return False

def keep_video(criteria,results):
    decision = False
    for i in my_criteria.index:
        print('checking ' + str(my_criteria.iloc[i,1]))
        if is_in_video(my_criteria.iloc[i,0],my_criteria.iloc[i,2],results_df):
            decision = True
            print(True)
        else:
            print(False)
    print('outcome: ' + str(decision))
    return decision

def delete_file(path):
    os.remove(path)
    print(path + ' file deleted')

# relocate a successful file process to output folder
def move_video(vid,new_dir):
    new_loc = new_dir + '//' + Path(vid).stem + '.mp4'
    Path(vid).rename(new_loc)
    print(vid + ' file moved to ' + new_loc)

def process_directory(chosen_model,chosen_source_directory,chosen_output_directory,chosen_frame_breaks,my_criteria):
	# run process

    input_details, output_details, interpreter = load_model(model_path=chosen_model)
    video_files = get_videos(chosen_source_directory)

    # run process for each video
    for i in video_files:
        video_to_process = i
        output_video_file = set_output_name(video_to_process,chosen_output_directory)
        video_capture, output_video = open_video_streams(video_to_process,output_video_file)
        results_df = process_video_with_ml(video_capture, interpreter, input_details, output_details, output_video, max_frame_break=chosen_frame_breaks)
        close_video_connections(video_capture,output_video)

        kv = keep_video(my_criteria,results_df)

        if not kv:
            delete_file(output_video_file)
            delete_file(video_to_process)
        else:
            move_video(video_to_process,chosen_output_directory)

    print('process complete')


