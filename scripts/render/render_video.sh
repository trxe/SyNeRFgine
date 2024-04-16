#!/bin/bash
set -e
FOLDER=$1
ffmpeg -framerate 24 -i "$FOLDER/output-%03d.png" -c:v libx264 -pix_fmt yuv420p "$FOLDER/out.mp4"
ffmpeg -i "$FOLDER/out.mp4" -vf "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 "$FOLDER/figure.gif"
