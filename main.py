# import the opencv library 
import cv2 
# import the numpy library as np
import numpy as np
# import the math library 
import math
#import traceback library 
import traceback

import wx
from pynput.mouse import Button, Controller

#display corresponding gestures which are in their ranges    
def count_number_of_finger(l,frame,areacnt,arearatio):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if l == 6:
        cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        pass

    if areacnt<1000:
        cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    
    else:
        if arearatio<12:
            cv2.putText(frame,str(l-1),(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame,str(l),(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

print("Opening Camera")
# define a video capture object 
vid = cv2.VideoCapture(0)


mouse = Controller()

app = wx.App(False)
(screenx,screeny) = wx.GetDisplaySize()
(capturex,capturey) = (700,400)#captures this size frame


vid.set(3,capturex)
vid.set(4,capturey)

kernelOpen = np.ones((5,5))#if noise are present other than yellow area
kernelClose = np.ones((20,20)) #if noise are present in yellow area

# range of the skin colour is defined
lower_skin = np.array([0,20,75], dtype=np.uint8)
upper_skin = np.array([45,255,255], dtype=np.uint8)

cd = 0


while(True): 

    try:  
        #an error comes if it does not find anything in window as it cannot find contour of max area
        #therefore this try error statement

        ret, frame = vid.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)

        #define roi which is a small square on screen
        roi=frame[100:400, 350:700]

        cv2.rectangle(frame,(350,100),(700,400),(0,255,0),0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        
    #extract skin colour image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)

    #image is blurred using GBlur
        mask = cv2.GaussianBlur(mask,(5,5),100)

    #find contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

   #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)

    #make convex hull around hand
        hull = cv2.convexHull(cnt)

    #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100

    #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

    # l = no. of defects
        l=0

    #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

            #distance between point and convex hull
            d=(2*ar)/a

            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)

            #draw lines around hand
            cv2.line(roi,start, end, [0,255,0], 2)
        
            west = tuple(cnt[cnt[:, :, 0].argmin()][0]) #gives the co-ordinate of the left extreme of contour
            east = tuple(cnt[cnt[:, :, 0].argmax()][0]) #above for east i.e right
            north = tuple(cnt[cnt[:, :, 1].argmin()][0])
            south = tuple(cnt[cnt[:, :, 1].argmax()][0])
            centre_x = (west[0]+east[0])/2
            centre_y = (north[0]+south[0])/2

            if (areacnt in range(8000,18000)): #hand is open
                mouse.release(Button.left)
                cv2.circle(frame, (int(centre_x),int(centre_y)), 6, (255,0,0), -1)#plots centre of the area
                mouse.position = (screenx-(centre_x*screenx/capturex),screeny-(centre_y*screeny/capturey))
                
            elif(areacnt in range(2000,7000)): #hand is closed
                cv2.circle(frame, (int(centre_x),int(centre_y)), 10, (255,255,255), -1)#plots centre of the area
                mouse.position = (screenx-(centre_x*screenx/capturex), screeny-(centre_y*screeny/capturey))
                mouse.press(Button.left)

        count_number_of_finger(l+1,frame,areacnt,arearatio)
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except:
        print(traceback.print_exc())
        pass

    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing Camera")
        break

# After the loop release the cap object 
vid.release() 

# Destroy all the windows 
cv2.destroyAllWindows() 
