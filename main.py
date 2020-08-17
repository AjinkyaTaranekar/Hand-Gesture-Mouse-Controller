# import the opencv library 
import cv2 
# import the numpy library as np
import numpy as np
# import the math library 
import math
#import traceback library 
import traceback

print("Opening Camera")
# define a video capture object 
vid = cv2.VideoCapture(0)

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



    # range of the skin colour is defined
        lower_skin = np.array([0,20,75], dtype=np.uint8)
        upper_skin = np.array([45,255,255], dtype=np.uint8)

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


        print("no of defects", l)
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
