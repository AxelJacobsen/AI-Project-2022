import os

class credentials(object):
  DATADIR = os.getcwd()+"\\dataset images"
  CATEGORIES = ["Closed_fist","Open_palm","Peace_sign","Pinky","Pointing","Rocknroll", "Spiderman", "Spock", "Thumbs_up"] 
  
  isConstantSize = True # sets if image has constants size
  # DOWNSIZE = 50 # current downsize if size is scalable, currently non-functioning
  IMGSIZE = 350 # constant size to put images as
  