#!/bin/bash

for file in /home/ikenna/IkennaWorkSpace/espressoTester/sample_images/*.jpg; do
    imgName=$(basename -- "$file")
    imgName="${imgName%%.*}"
    ./espressoTester $file $PWD/sample_img_results/"$imgName"_detection.jpg
done
