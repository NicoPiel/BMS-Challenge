import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def fill_rings(im):

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    vcnts, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.ones(gray.shape, np.uint8)

    for cnt in contours:
        cv2.drawContours(canvas, [cnt], -1, (0, 0, 255),-1)

    # remove the contours from the image and show the resulting images
    img = cv2.bitwise_and(im, im, mask=canvas)

    return img


def get_ring_coordinates(img):
    
    kernel = np.ones((14,14), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    vcnts, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    canvas2 = np.ones(img.shape, np.uint8)

    rings = []
    for cnt in contours:
        for i in range(1,5):
            p = i/100
            approx = cv2.approxPolyDP(cnt,p*cv2.arcLength(cnt,True),True)
            if len(approx)==6:
                cv2.drawContours(canvas2, [cnt], -1, (255, 255, 255),-1)
                rings.append(cnt)
                break

    return rings, canvas2


def get_double_bond_image(img):
    
    kernel = np.ones((10,10),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1) 
    kernel = np.ones((12,12),np.uint8) 
    img = cv2.dilate(img,kernel,iterations = 1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    vcnts, thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    mask = np.ones(img.shape, np.uint8)

    for cnt in contours:
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255),-1)
        
    mask = 255-mask
    return mask
    
    
def get_d_bond_and_letter_coordinates(mask):
    
    font = cv2.FONT_HERSHEY_COMPLEX
    img = mask

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    vcnts, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    w_sum = 0.0
    h_sum = 0.0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        (x, y, w, h) = cv2.boundingRect(approx)
        w_sum += w
        h_sum += h
        i+=1

    w_avg = w_sum/i
    h_avg = h_sum/i

    letters = []
    d_bond = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (0), 1)

        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # No important point
        if w <= 1.2*w_avg and w <= 0.1*w_avg and h <= 1.2*h_avg and h <= 0.1*h_avg:
            continue

        # Is a letter
        if w <= 1.2*w_avg and h <= 1.2*h_avg:
            cv2.putText(img, "z", (x, y), font, 1, (0))
            letters.append(cnt)

        # is a double bond
        else:
            cv2.putText(img, "b", (x, y), font, 1, (0))
            d_bond.append(cnt)
    
    return img, letters, d_bond
