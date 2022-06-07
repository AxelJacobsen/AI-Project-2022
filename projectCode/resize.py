#!/usr/bin/python
from PIL import Image
import os
from constants.credentials import credentials as CR

path = "./"+CR.FOLDR[1]+"/"
savePath = "./"+CR.FOLDR[0]+"/"

imgSizeTo = CR.IMGRESIZETO

def resize(saveFrom, saveTo, size):
    dirs = os.listdir( saveFrom )
    for item in dirs:
        if os.path.isfile(saveFrom+item):
            im = Image.open(saveFrom+item)
            f, e = os.path.splitext(saveTo+item)
            imResize = im.resize((size,size), Image.Resampling.LANCZOS).rotate(270)
            imResize.save(f + '_resized.jpg', 'JPEG', quality=90)
    print("RESIZEDATA::done!")


for foldr in CR.CATEGORIES:
  print("RESIZEDATA::resizing for "+foldr+"...")
  dataFrom = path + foldr +"/"
  dataTo = savePath + foldr +"/"
  resize(dataFrom, dataTo, imgSizeTo)