class credentials(object):
  PATH = "./"
  FOLDR = ["resizedData","originalData"]
  DIR = PATH + FOLDR[0]
  CATEGORIES = ['Closed_fist', 'Finger_guns', 'Open_palm', 'Peace_sign', 'Pinky', 'Pointing', 'Rocknroll', 'Spiderman', 'Spock', 'Thumbs_up']
  IMGRESIZETO = 500 # constant size to resize from original data
  IMGSIZE = 250 # constant size to put images as