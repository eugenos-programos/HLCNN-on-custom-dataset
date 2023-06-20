from roboflow import Roboflow



rf = Roboflow(api_key="OHFxIoh14dlz0sCSILVz")
project = rf.workspace("boxes-dataset-minmax").project("boxes-updated")
dataset = project.version(22).download("yolov7")
