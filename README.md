### Setup

The best way to setup and check you can run all the code in this repo is to run through the steps in the notebook `process_directory_walkthrough.ipynb`, which includes steps for installing libraries. 

### Setting parameters

The parameters are set in the first section of the main.py script. 

`chosen_model`: this is set to `yolo-v5.tflite`. I haven't updated code to accept other model variants. Check that the `yolo-v5.tflite` is present in your main path. 
`chosen_source_directory`: Set this to the path of the directory you want to process
`chosen_output_directory`: Set this to the path of the directory you want to save outputs. 
`chosen_frame_breaks`: Set this to the number of video frames before you kill the loop and move on to the next video. Set a high value if you want to ensure all of a video is processed.  
d = {'class': [15, 77],'class_name': ['cat','teddy bear'] , 'threshold': [0.5, 0.5]} # see list here: https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
`my_criteria`: This needs to be a dataframe of the form `{'class': [1, 2, 3],'class_name': ['a','b','c'] , 'threshold': [0.25,0.5,0.75]`. You can get a list of objects that the model will accept from [here](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml). 

### Execution

In termimal execute `python main.py` to run the process. 