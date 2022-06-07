#!/usr/bin/python
from PIL import Image
import os, sys
from constants.credentials import credentials as CR

path = "./originalData/"
savePath = "./resizedData/"

imgSizeTo = 500

def resize(saveFrom, saveTo, size):
    dirs = os.listdir( saveFrom )
    for item in dirs:
        if os.path.isfile(saveFrom+item):
            im = Image.open(saveFrom+item)
            f, e = os.path.splitext(saveTo+item)
            imResize = im.resize((size,size), Image.ANTIALIAS).rotate(270)
            imResize.save(f + '_resized.jpg', 'JPEG', quality=90)


for foldr in CR.CATEGORIES:
  dataFrom = path + foldr +"/"
  dataTo = savePath + foldr +"/"
  resize(dataFrom, dataTo, imgSizeTo)