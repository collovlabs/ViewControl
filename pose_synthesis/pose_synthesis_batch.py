import os
import subprocess
import argparse
from multiprocessing import Pool

def generate_images(args):
    x, y, img_path, half_precision, output_path = args
    x = "["+x+"]"
    y = "["+y+"]"

    command = f"python pose_synthesis.py --x {x} --y {y} --img_path {img_path} --half_precision --output_path {output_path}"
    subprocess.run(command, shell=True)

def main(input_dir, output_dir, x_values, y_values):
    if not os.path.isdir(input_dir):
        image_files = []
    else:
        image_files = os.listdir(input_dir)
    
    # if input_dir is a single image
    if len(image_files) == 0:
        image_files = [input_dir.split('/')[-1]]
    
    for image_file in image_files:
        print("Processing image: ", image_file)
        image_name = os.path.splitext(image_file)[0]
        if len(image_files) == 1:
            image_path = input_dir
        else:
            image_path = os.path.join(input_dir, image_file)

        output_subdir = os.path.join(output_dir, image_name)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        # x_values = [-10,0,10]
        # y_values = [0,-10,-20, -30, -40,-50,10,20,30,40,50,60]

        args_list = []
        args_list.append((x_values, y_values, image_path, True, output_subdir))

        # use multiple threads to generate images
        with Pool(processes=2) as pool:
            pool.map(generate_images, args_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./input_images_dir")
    parser.add_argument("--output_dir", type=str, default="./output_images_dir")
    parser.add_argument("--x_values", type=str, default="0,10", help="comma separated list of x values")
    parser.add_argument("--y_values", type=str, default="0,-10", help="comma separated list of y values")
    parser.add_argument("--pose_file_path", type=str, default=None)

    args = parser.parse_args()

    if args.pose_file_path:
        with open(args.pose_file_path, "r") as f:
            args.x_values, args.y_values = f.read().split(' ')
    main(args.input_dir, args.output_dir, args.x_values, args.y_values)
