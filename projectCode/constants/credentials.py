import os

class credentials(object):
  FOLDR = ["dataset images", "../originalData"]
  DATADIR = os.getcwd()+"\\"+ FOLDR[1]
  CATEGORIES = ['Closed_fist', 'Finger_guns', 'Open_palm', 'Peace_sign', 'Pinky', 'Pointing', 'Rocknroll', 'Spiderman', 'Spock', 'Thumbs_up']
  
  isConstantSize = True # sets if image has constants size
  # DOWNSIZE = 50 # current downsize if size is scalable, currently non-functioning
  IMGSIZE = 100 # constant size to put images as
  