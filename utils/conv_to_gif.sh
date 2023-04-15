#!/bin/bash

# Script to convert movie files to a gif

if [ $# -eq 0 ]
  then
    echo "Input media file is missing!" 
    echo "Usage: bash conv_to_gif.sh <movie_file> "
  else
    echo "Argument provided $1"
    ffmpeg -i $1 -r 25 output.gif
fi

