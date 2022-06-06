import os

class credentials(object):
  FOLDR = "originalData"
  DATADIR = os.getcwd()+"\\"+ FOLDR[1]
  CATEGORIES = ['Closed_fist', 'Finger_guns', 'Open_palm', 'Peace_sign', 'Pinky', 'Pointing', 'Rocknroll', 'Spiderman', 'Spock', 'Thumbs_up']
  
  IMGSIZE = 128 # constant size to put images as
  