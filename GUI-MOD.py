# GUI modded by MoonDragon - V.2024-08-23 https://github.com/MoonDragon-MD/
import numpy as np
import cv2
import PySimpleGUI as sg
import os
import argparse
import sys
import shutil
from subprocess import call

def modify(image_filename=None, output_folder=None, gpu="0", hr=False, with_scratch=False, window=None):
    def run_cmd(command):
        """Execute a shell command and handle interruptions."""
        try:
            call(command, shell=True)
        except KeyboardInterrupt:
            print("Process interrupted")
            sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=image_filename, help="Test images")
    parser.add_argument("--output_folder", type=str, default=output_folder, help="Restored images, please use the absolute path.")
    parser.add_argument("--GPU", type=str, default=gpu, help="Select GPU id (0, 1, 2)")
    parser.add_argument("--checkpoint_name", type=str, default="Setting_9_epoch_100", help="Choose checkpoint")
    parser.add_argument("--HR", action="store_true" if hr else "store_false", help="High Resolution")
    parser.add_argument("--with_scratch", action="store_true" if with_scratch else "store_false", help="Image with scratch")
    opts = parser.parse_args()

    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    os.makedirs(opts.output_folder, exist_ok=True)

# Check if input is a file or a directory
    if os.path.isfile(opts.input_folder):
        input_dir = os.path.dirname(opts.input_folder)
        temp_dir = os.path.join(input_dir, "temp_input")
        os.makedirs(temp_dir, exist_ok=True)
        shutil.copy(opts.input_folder, temp_dir)
        opts.input_folder = temp_dir

# Stage 1: Overall Quality Improve
    window.write_event_value('-UPDATE-', "Running Stage 1: Overall restoration")
    os.chdir("./Global")
    stage_1_input_dir = opts.input_folder
    stage_1_output_dir = os.path.join(opts.output_folder, "stage_1_restore_output")
    os.makedirs(stage_1_output_dir, exist_ok=True)

    if not opts.with_scratch:
        stage_1_command = (
            "python test.py --test_mode Full --Quality_restore --test_input "
            + stage_1_input_dir
            + " --outputs_dir "
            + stage_1_output_dir
            + " --gpu_ids "
            + opts.GPU
        )
        if opts.HR:
            stage_1_command += " --HR"
        run_cmd(stage_1_command)
    else:
        mask_dir = os.path.join(stage_1_output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")

        stage_1_command_1 = (
            "python detection.py --test_path " + stage_1_input_dir +
            " --output_dir " + mask_dir +
            " --input_size full_size" +
            " --GPU " + opts.GPU
        )
        run_cmd(stage_1_command_1)

        stage_1_command_2 = (
            "python test.py --Scratch_and_Quality_restore --test_input "
            + new_input +
            " --test_mask " + new_mask +
            " --outputs_dir " + stage_1_output_dir +
            " --gpu_ids " + opts.GPU
        )
        if opts.HR:
            stage_1_command_2 += " --HR"
        run_cmd(stage_1_command_2)

    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    final_output_dir = os.path.join(opts.output_folder, "final_output")
    os.makedirs(final_output_dir, exist_ok=True)
    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, final_output_dir)

    window.write_event_value('-UPDATE-', "Finish Stage 1 ...")

# Stage 2: Face Detection
    window.write_event_value('-UPDATE-', "Running Stage 2: Face Detection")
    os.chdir("../Face_Detection")
    stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_2_output_dir = os.path.join(opts.output_folder, "stage_2_detection_output")
    os.makedirs(stage_2_output_dir, exist_ok=True)

    stage_2_command = (
        "python detect_all_dlib.py --url " + stage_2_input_dir +
        " --save_url " + stage_2_output_dir
    )
    run_cmd(stage_2_command)
    window.write_event_value('-UPDATE-', "Finish Stage 2 ...")

# Stage 3: Face Enhance
    window.write_event_value('-UPDATE-', "Running Stage 3: Face Enhancement")
    os.chdir("../Face_Enhancement")
    stage_3_output_dir = os.path.join(opts.output_folder, "stage_3_face_output")
    os.makedirs(stage_3_output_dir, exist_ok=True)

    stage_3_command = (
        "python test_face.py --old_face_folder " + stage_2_output_dir +
        " --old_face_label_folder ./ --tensorboard_log --name " +
        opts.checkpoint_name + " --gpu_ids " + opts.GPU +
        " --load_size 256 --label_nc 18 --no_instance " +
        "--preprocess_mode resize --batchSize 4 --results_dir " +
        stage_3_output_dir + " --no_parsing_map"
    )
    run_cmd(stage_3_command)
    window.write_event_value('-UPDATE-', "Finish Stage 3 ...")

# Stage 4: Warp back
    window.write_event_value('-UPDATE-', "Running Stage 4: Blending")
    os.chdir("../Face_Detection")
    stage_4_input_image_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")

    stage_4_command = (
        "python align_warp_back_multiple_dlib.py --origin_url " +
        stage_4_input_image_dir + " --replace_url " +
        stage_4_input_face_dir + " --save_url " + final_output_dir
    )
    run_cmd(stage_4_command)
    window.write_event_value('-UPDATE-', "Finish Stage 4 ...")

    window.write_event_value('-UPDATE-', "All processing is done. Please check the results.")

# GUI Definition
layout = [
    [sg.Text("Select the image or folder to modify"), sg.Input(), sg.FileBrowse()],
    [sg.Text("Select the output folder"), sg.Input(), sg.FolderBrowse(key="-OUTPUT-")],
    [sg.Text("Select GPU"), sg.Combo(["0", "1", "2"], default_value="0", key="-GPU-")],
    [sg.Checkbox("High-Resolution Image", key="-HR-")],
    [sg.Checkbox("Image with scratches", key="-SCRATCH-")],
    [sg.Button("Modify"), sg.Button("Exit")],
    [sg.Text("", key="-STATUS-", size=(40, 1))]  # Text field to show processing status
]

# Create the window
window = sg.Window("Bringing Old Photos Back-to Life", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    elif event == "Modify":
        image_filename = values[0]
        output_folder = values["-OUTPUT-"]
        gpu = values["-GPU-"]
        hr = values["-HR-"]
        with_scratch = values["-SCRATCH-"]
        modify(image_filename=image_filename, output_folder=output_folder, gpu=gpu, hr=hr, with_scratch=with_scratch, window=window)
    elif event == '-UPDATE-':
        window['-STATUS-'].update(values[event])  # Update the status text with messages

window.close()
