#!/usr/bin/python
from PIL import Image
import os, sys

path = "./originalData/Closed_fist/"
dirs = os.listdir( path )
savePath = "./originalData/testResult/"

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(savePath+item)
            imResize = im.resize((500,500), Image.ANTIALIAS).rotate(270)
            imResize.save(f + '_resized.jpg', 'JPEG', quality=90)

resize()