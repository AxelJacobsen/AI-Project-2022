import cv2
import os
import numpy as np
from constants.credentials import credentials
import pickle
import random as rd

class handleTrainingData ():
  CATEGORIES = [] 
  DATADIR = ""
  IMGSIZE = 0 
  
  Data = []
  X = [] # feature set
  y = [] # labels
  
  
  def __init__(self, c,d,i):
    self.CATEGORIES = c
    self.DATADIR = d
    self.IMGSIZE = i
  
  def shuffle(self,data):
    """Shuffles data

    Args:
        data (_type_): data to shuffle
    """
    rd.shuffle(data)
    

  def importData(self, val):
    # Import current set
    if val.lower() == "x" :
      pickle_in = open("X.pickle","rb")
      self.X = pickle.load(pickle_in)
    elif val.lower() == "y" :
      pickle_in = open("y.pickle","rb")
      self.y = pickle.load(pickle_in)




  def exportData(self, X, y):
    # Export current set
    pickle_out = open("X.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

  def init_data(self):
    """Loads and catecorizes training data
    """
    for category in self.CATEGORIES:
      path = os.path.join(self.DATADIR,category)
      class_num = category
      for get_img in os.listdir(path):
        try:
          raw_img = cv2.imread(os.path.join(path,get_img),cv2.IMREAD_GRAYSCALE) # get images as grayscale
          if credentials.isConstantSize:
            dim = (credentials.IMGSIZE, credentials.IMGSIZE)  # gets constant dimensions
            """else: # CURRENTLY NON FUNCTIONING 
            dim = (int((raw_img.shape [0]*credentials.DOWNSIZE)/100), int((raw_img.shape [1]*credentials.DOWNSIZE)/100)) # gets resized dimensions """ 
          img = cv2.resize(raw_img, dim, interpolation=cv2.INTER_AREA) # resizes the image
          self.Data.append([img, class_num])
        except Exception as e: # in case of invalid images
          pass # normally we should throw an exception, however temp disabled.
        
        
  def append(self):
    for features, label in self.Data:
      self.X.append(features)
      self.y.append(label)
    self.X = np.array(self.X).reshape(-1, self.IMGSIZE, self.IMGSIZE, 1)
