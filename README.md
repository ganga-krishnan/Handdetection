# Handdetection
The Python code consists of Train object detection, ensembled with a classifier using the below link (to increase the accuracy
between classes). The inference script can take a rtsp video and give an inference. With FPS printed on-screen on
video.MobileNet as a backbone should be used.<br>The task uses a dataset with labeled images of hands available in the official website of Roboflow.<br>
<br>Dataset link: https://universe.roboflow.com/handdetection/hand_detection-cwgzh<br><br>

## Steps
1. Clone this repository: https://github.com/ganga-krishnan/Handdetection <br><br>
2. Create a new virtual environment<br>
>_python -m venv tfod_ <br><br>
3. Activate your virtual environment<br>
>_source tfod/bin/activate # Linux_<br>
>_.\tfod\Scripts\activate # Windows_ <br><br>
4.  Install dependencies and add virtual environment to the Python Kernel<br>
>_python -m pip install --upgrade pip_ <br>
>_pip install ipykernel_ <br>
>_python -m ipykernel install --user --name=tfodj_ <br><br>
5. Begin training process by opening 2. Training and Detection.ipynb, this notebook will walk you through installing Tensorflow Object Detection, making detections, saving and exporting your model.<br><br>
6.  During this process the Notebook will install Tensorflow Object Detection. You should ideally receive a notification indicating that the API has installed successfully at Step 6 with the last line stating OK.<br><br>
If not, resolve installation errors by referring to the Error Guide.md in this folder.<br><br>
7. Train the model, inside of the notebook, you may choose to train the model from within the notebook. I have noticed however that training inside of a separate terminal on a Windows machine you're able to display live loss metrics.<br><br>
## Resources
* wget.download('https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py')<br>
* Setup https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html
* Tensorflow Models: https://github.com/tensorflow/models
* 
