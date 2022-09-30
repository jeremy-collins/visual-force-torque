from multiprocessing import synchronize
import re
import cv2
import sys
import numpy as np
import time
from datetime import datetime
import os
from recording.gripper_camera import Camera
from recording.ft_stretch_v1 import FTCapture
# from recording.move_gripper import Gripper
import argparse
import threading
# import keyboard
import robot.zmq_server as zmq_server
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from prediction.config_utils import *
import json

from __future__ import print_function


import stretch_body.self.xbox_controller as xc
import stretch_body.robot as rb
from stretch_body.hello_utils import *
from robot.stretch_self.xbox_controller_teleop import *
import os
import time
import argparse
print_stretch_re_use()

parser=argparse.ArgumentParser(description=
     'Jog the robot from an XBox Controller  \n' +
     '-------------------------------------\n' +
    'Left Stick X:\t Rotate base \n' +
    'Left Stick Y:\t Translate base \n' +
    'Right Trigger:\t Fast base motion \n' +
    'Right Stick X:\t Translate arm \n' +
    'Right Stick Y:\t Translate lift \n' +
    'Left Button:\t Rotate wrist CCW \n' +
    'Right Button:\t Rotate wrist CW \n' +
    'A/B Buttons:\t Close/Open gripper \n' +
    'Left/Right Pad:\t Head Pan \n' +
    'Top/Bottom Pad:\t Head tilt \n' +
    'Y Button :\t Go to stow position \n ' +
    'Start Button:\t Home robot \n ' +
    'Back Button (2 sec):\t Shutdown computer \n ' +
    '-------------------------------------\n',formatter_class=argparse.RawTextHelpFormatter)

args=parser.parse_args()

class DataCapture:
    def __init__(self):
        _, self.args = parse_config_args()
        resource = 3 if self.args.on_robot else 0
        self.feed = Camera(resource=resource, view=self.args.view)
        self.ft = FTCapture()
        self.delta_lin = 0.02
        self.delta_ang = 0.1
        self.enable_moving = True

        # if self.args.robot_state:
        #     self.client = zmq_client.SocketThreadedClient(ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER)
        #     self.server = zmq_server.SocketServer(port=zmq_client.PORT_COMMAND_SERVER)

        print("ROBOT STATE: ", self.args.robot_state)

        # counting the number of folders in the stage folder beginning with args.folder
        folders = os.listdir(os.path.join('data', self.args.stage))
        
        if len(folders) == 0:
            folder_count = 0
        else:
            folder_count = len([f for f in folders if re.match(self.args.folder, f)])

        self.args.folder = self.args.folder + '_' + str(folder_count)

        # naming the folders where the data will be saved
        self.image_folder = os.path.join('data', self.args.stage, self.args.folder, 'cam')
        self.ft_folder = os.path.join('data', self.args.stage, self.args.folder,'ft')
        self.state_folder = os.path.join('data', self.args.stage, self.args.folder,'robot_state')
        
        # making directories for data if they doesn't exist
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        if not os.path.exists(self.ft_folder):
            os.makedirs(self.ft_folder) 
        if not os.path.exists(self.state_folder):
            os.makedirs(self.state_folder)

        # xbox teleop
        global use_head_mapping, use_dex_wrist_mapping, use_stretch_gripper_mapping
        self.xbox_controller = xc.XboxController()
        self.xbox_controller.start()
        self.robot = rb.Robot()

        self.robot.startup()
        print('Using key mapping for tool: %s'%self.robot.end_of_arm.name)
        if self.robot.end_of_arm.name=='tool_none':
            use_head_mapping=True
            use_stretch_gripper_mapping=False
            use_dex_wrist_mapping=False

        if self.robot.end_of_arm.name=='tool_stretch_gripper':
            use_head_mapping=True
            use_stretch_gripper_mapping=True
            use_dex_wrist_mapping=False

        if self.robot.end_of_arm.name=='tool_stretch_dex_wrist':
            use_head_mapping=False
            use_stretch_gripper_mapping=True
            use_dex_wrist_mapping=True

        self.robot.pimu.trigger_beep()
        self.robot.push_command()
        time.sleep(0.5)

        self.robot.pimu.trigger_beep()
        self.robot.push_command()
        time.sleep(0.5)

    def capture_data(self):
        # get data snapshots
        frame = self.feed.get_frame()
        ft_data = self.ft.get_ft()

        if self.args.robot_state:
            robot_ok, self.pos_dict = read_robot_status(self.client)
        else:
            self.pos_dict = None

        # set file names based on timestamps
        image_name = str(self.feed.current_frame_time) + '.jpg'
        ft_name = str(self.ft.current_frame_time)
        state_name = str(self.ft.current_frame_time) + '.txt'

        # save data to machine
        if self.args.stage in ['train', 'test', 'raw']:
            cv2.imwrite(os.path.join(self.image_folder, image_name), frame)
            np.save(os.path.join(self.ft_folder, ft_name), ft_data)

            with open(os.path.join(self.state_folder, state_name), 'w') as convert_file:
                if self.args.robot_state:
                    convert_file.write(json.dumps(self.pos_dict))
                else:
                    convert_file.write(json.dumps({"no_robot_state":None}))

        else:
            print('Invalid stage argument. Please choose train, test, or raw.')
            sys.exit(1)

        result = {'frame':frame, 'frame_time':self.feed.current_frame_time, 'ft_frame':ft_data, 'ft_frame_time':self.ft.current_frame_time, 'robot_state':self.pos_dict}
        
        if self.feed.view:
            cv2.imshow("frames", frame)
        if self.args.robot_state:
            self.xbox_teleop()
        else:
            cv2.waitKey(1)


        print('Average FPS', self.feed.frame_count / (time.time() - self.feed.first_frame_time))

        return result

    def xbox_teleop(self):
        controller_state = self.xbox_controller.get_state()
        if not self.robot.is_calibrated():
            manage_calibration(self.robot, controller_state)
        else:
            manage_base(self.robot, controller_state)
            manage_lift_arm(self.robot, controller_state)
            manage_end_of_arm(self.robot, controller_state)
            manage_head(self.robot, controller_state)
            manage_stow(self.robot, controller_state)
        manage_shutdown(self.robot, controller_state)
        self.robot.push_command()

if __name__ == "__main__":
    cap = DataCapture()
    delay = []

    for i in range(4500):
        data = cap.capture_data()
        delay.append(data['ft_frame_time'] - data['frame_time'])
        
    print('saved results to {}'.format(os.path.join('data', cap.args.stage, cap.args.folder)))
    print("delay avg:", np.mean(delay))
    print("delay std:", np.std(delay))