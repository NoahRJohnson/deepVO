#!/usr/bin/env python3
""""Downloads and unzips KITTI odometry benchmark video for odometry.
Warning: This can take a while, and uses up ~ 65Gb of disk space."""

import argparse
import os
import sys

from subprocess import call

URL_BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti/"
zip_tags = ['data_odometry_color', 'data_odometry_poses']

def main():

    # Read in where to place this data we're downloading
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', type=str, help='folder to place data')
    args = parser.parse_args()
    out_dir = args.outdir

    # Make the output directory if it doesn't already exist
    os.makedirs(out_dir, exist_ok=True)
    # Move inside that directory
    os.chdir(out_dir)

    # Append .zip to every file name to download
    zip_names = [name + ".zip" for name in zip_tags]

    for zip_name in zip_names:

        if os.path.exists(zip_name):
            print("File {} exists. Not re-downloading.".format(zip_name))
            continue

        # else download zip
        url = URL_BASE + zip_name
        print("Downloading file \"{}\" to folder \"{}\".".format(zip_name, out_dir))
        call(['wget', url])

        # and unzip it
        print("Unzipping file \"{}\" in folder \"{}\".".format(zip_name, out_dir))
        call(['unzip', zip_name])

    return 0

if __name__ == '__main__':
    sys.exit(main())
