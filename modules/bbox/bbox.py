import cv2
from random import randint

class BoundingBox:
    def __init__(self, x, y, w, h, id=0):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cx = int((x+x+w)/2)
        self.cy = int((y+y+h)/2)
        self.c = (randint(0,255), randint(0,255), randint(0,255))
        self.name = None
        self.emotion = None

    def area(self):
        return self.w * self.h
    
    def draw(self, frame):
        return cv2.rectangle(frame, (self.x, self.y), (self.x+self.w, self.y+self.h), self.c, 2)
    
    def distance(self, other, distance_metric="manhattan"):
        if distance_metric=="manhattan":
            return abs(self.cx-other.cx)+abs(self.cy-other.cy)
        else:
            return 0