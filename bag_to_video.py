import io
import numpy as np
from PIL import Image

import os
from rosbag import Bag
from tqdm import tqdm
import cv2

def numpy_image(data: any, dims: tuple) -> np.ndarray:
    """
    Return a NumPy tensor from image data.

    Args:
        data: the image data to convert to a NumPy tensor
        dims: the height and width of the image

    Returns:
        an RGB NumPy tensor of the image data

    """
    # Convert BGR image to RGB
    return np.array(data, dtype='uint8').reshape((*dims, 3))[:, :, ::-1]
    

def get_camera_image(data: bytes, dims: tuple) -> np.ndarray:
    """
    Get an image from binary ROS data.

    Args:
        data: the binary data to extract an image from
        dims: the expected dimensions of the image

    Returns:
        an uncompressed NumPy tensor with the 8-bit RGB pixel data

    """
    try:
        # open the compressed image using Pillow
        with Image.open(io.BytesIO(data)) as rgb_image:
            return numpy_image(rgb_image, dims)
    # if an OS error happens, the image is raw data
    except OSError:
        return numpy_image(list(data), dims)


def bag_to_video(
    input_file: Bag,
    output_file: str,
    topic: str,
    fps: int=30,
    codec: str='mp4v',
    ) -> None:
    """
    Convert a ROS bag with image topic to a video file.

    Args:
        input_file: the bag file to get image data from
        output_file: the path to an output video file to create
        topic: the topic to read image data from
        fps: the frame rate of the video
        codec: the codec to use when outputting to the video file

    Returns:
        None
    """
    # create an empty reference for the output video file
    video = None
    # get the total number of frames to write
    total = input_file.get_message_count(topic_filters=topic)
    # get an iterator for the topic with the frame data
    iterator = input_file.read_messages(topics=topic)
    # iterate over the image messages of the given topic
    for _, msg, _ in tqdm(iterator, total=total):
        # open the video file if it isn't open
        if video is None:
            # create the video codec
            codec = cv2.VideoWriter_fourcc(*codec)
            # open the output video file
            cv_dims = (msg.width, msg.height)
            video = cv2.VideoWriter(output_file, codec, fps, cv_dims)
        # read the image data into a NumPy tensor
        img = get_camera_image(msg.data, (msg.height, msg.width))
        # write the image to the video file
        video.write(img)

    # if the video file is open, close it
    if video is not None:
        video.release()

    # read the start time of the video from the bag file
    start = input_file.get_start_time()
    # set the access and modification times for the video file
    os.utime(output_file, (start, start))


def process_bag_files(
        input_folder: str, 
        output_folder: str,
        topic: str, 
        codec: str='mp4v'
        )-> None:
    """
    Process each .bag file in the input folder and save it as a video in the output folder.

    Args:
        input_folder: Path to the folder containing .bag files
        output_folder: Path to the folder where videos will be saved
        codec: Codec to use when outputting to the video file

    Returns:
        None
    """
    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".bag"):
            # Construct full paths for input and output files
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "." + codec)

            if not os.path.exists(output_file_path):
                # Process the current bag file
                bag = Bag(input_file_path)
                bag_to_video(bag, output_file_path, topic, codec=codec)
            else: 
                print("File already converted")


def get_img_from_bag(
    input_file: Bag,
    output_file: str,
    topic: str,
    frame_number: int=0,
    ) -> None:
    """
    Get an image from a ROS bag with image topic.

    Args:
        input_file: the bag file to get image data from
        output_file: the path to an output image file to create
        topic: the topic to read image data from
        frame_number: the frame to save as an image
    Returns:
        None
    """

    #create an empty reference for the output image file 
    img = None 
    # get the total number of frames to write
    total = input_file.get_message_count(topic_filters=topic)
    
    # make sure that the selected frame is in range
    if frame_number < 0: 
        frame_number = 0
    elif frame_number > total -1: 
        frame_number = total -1
    # get an iterator for the topic with the frame data
    iterator = input_file.read_messages(topics=topic)

    # iterate over the image messages of the given topic until the wanted frame is reached
    for index, msg in enumerate(iterator): 
        if index == frame_number: 
            #get the message data from the rosbag message 
            msg_data = msg.message
            # read the image data into a NumPy tensor
            img = get_camera_image(msg_data.data, (msg_data.height, msg_data.width))
            #save the image
            cv2.imwrite(output_file, img)
            print(f"Saved img at: {output_file}")
            return



if __name__ == "__main__":
    #define the topic that contains the image data
    topic = "/device_0/sensor_1/Color_0/image/data"

    #get image from bag file
    file_path = "maze_setup_initial.bag"
    output_file_path = "maze_setup_initial.png"
    bag = Bag(file_path)
    get_img_from_bag(input_file=bag, output_file=output_file_path, topic=topic, frame_number=0)
    
    #get video from bag file 
    file_path = "test_recording.bag"
    output_file_path = "test_recording.mp4v"
    bag = Bag(file_path)
    bag_to_video(bag, output_file_path, topic)

    #convert each .bag file in folder 
    input_folder = "Experiment Recordings\\Bag Files"
    output_folder = "Experiment Recordings\\"
    process_bag_files(input_folder=input_folder, output_folder=output_folder, topic=topic, codec='mp4v')
