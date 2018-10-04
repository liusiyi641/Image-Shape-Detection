import numpy as np
import cv2
import collections
import operator
import copy
from matplotlib import pyplot as plt


#img = cv2.imread('shapes_train2018/1000.jpeg')
img = cv2.imread('shapes_train2018/1055.jpeg')
h = img.shape[0]
w = img.shape[1]
egdeList = []
BackgroundEgdes = []
SelfEdges = []
CuttingEdges = []

def denoising(img):
    return cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

def kmeans(img, K):
    
    Z = dst.reshape((-1,3))
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((dst.shape))
    return res2


def countPixel(img):
    h = img.shape[0]
    w = img.shape[1]
    my_dict = {}
    count = 0
    a = []
    for i in range(h):
        for j in range(w):
            count += 1
            b = img[i,j]
            temp = []
            for l in range(3):
                temp += [b[l]]
            a.append(temp)
    for i in a:
            i = str(i)
            if i in my_dict:
                my_dict[i] += 1
            else:
                my_dict[i] = 1
    sorted_dict = sorted(my_dict.items(), key=operator.itemgetter(1))
    return sorted_dict

def findBackgroundContour(image, kgraph, backgroundcolor):
    h = image.shape[0]
    w = image.shape[1]
    
    for i in range(w):
        if (not(kgraph[0,i][0] == int(backgroundcolor[0]) and kgraph[0,i][1] == int(backgroundcolor[1]) and kgraph[0,i][2] == int(backgroundcolor[2]))):
            #print("we have top line pixels")
            #image[0,i][0] = 0
            #image[0,i][1] = 0
            #image[0,i][2] = 0
            BackgroundEgdes.append([0,i])
    for i in range(w):
        if (not(kgraph[h-1,i][0] == int(backgroundcolor[0]) and kgraph[h-1,i][1] == int(backgroundcolor[1]) and kgraph[h-1,i][2] == int(backgroundcolor[2]))):
            #print("we have bot line pixels")
            #image[h-1,i][0] = 0
            #image[h-1,i][1] = 0
            #image[h-1,i][2] = 0
            BackgroundEgdes.append([h-1,i])
    for i in range(h):
        if (not(kgraph[i,0][0] == int(backgroundcolor[0]) and kgraph[i,0][1] == int(backgroundcolor[1]) and kgraph[i,0][2] == int(backgroundcolor[2]))):
            #print("we have left line pixels")
            #image[0,i][0] = 0
            #image[0,i][1] = 0
            #image[0,i][2] = 0 
            BackgroundEgdes.append([i,0])
    for i in range(h):
        if (not(kgraph[i,w-1][0] == int(backgroundcolor[0]) and kgraph[i,h-1][1] == int(backgroundcolor[1]) and kgraph[i,h-1][2] == int(backgroundcolor[2]))):
            #print("we have right line pixels")
            #image[i,w-1][0] = 0
            #image[i,w-1][1] = 0
            #image[i,w-1][2] = 0
            BackgroundEgdes.append([i,w-1])
    return

def Binarize(Color, res):

    copyImage = copy.copy(res)
    tempX = 0
    tempY = 0
    for i in range(h):
        for j in range(w):
            if (copyImage[i,j][0] != int(Color[0]) and copyImage[i,j][1] != int(Color[1]) and copyImage[i,j][2] != int(Color[2])):
                copyImage[i,j][0] = 0
                copyImage[i,j][1] = 0
                copyImage[i,j][2] = 0
    #print(copyImage[5,64])
    for i in range(h):
        for j in range(w):
            #print(i,j)
            if (copyImage[i,j][0] == int(Color[0]) and copyImage[i,j][1] == int(Color[1]) and copyImage[i,j][2] == int(Color[2])):
                tempX = i
                tempY = j
                
                if (i == 0 and j == 0):
                    if (
                        (copyImage[i+1,j+1][0] == int(Color[0]) and
                        copyImage[i+1,j+1][1] == int(Color[1]) and
                        copyImage[i+1,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i,j+1][0] == int(Color[0]) and
                        copyImage[i,j+1][1] == int(Color[1]) and
                        copyImage[i,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i+1,j][0] == int(Color[0]) and
                        copyImage[i+1,j][1] == int(Color[1]) and
                        copyImage[i+1,j][2] == int(Color[2]))
                    ):
                        return [i,j]
                    
                if (i == 0 and j == w-1):
                    if (
                        (copyImage[i+1,j-1][0] == int(Color[0]) and
                        copyImage[i+1,j-1][1] == int(Color[1]) and
                        copyImage[i+1,j-1][2] == int(Color[2]))
                        or
                        (copyImage[i,j-1][0] == int(Color[0]) and
                        copyImage[i,j-1][1] == int(Color[1]) and
                        copyImage[i,j-1][2] == int(Color[2]))
                        or
                        (copyImage[i+1,j][0] == int(Color[0]) and
                        copyImage[i+1,j][1] == int(Color[1]) and
                        copyImage[i+1,j][2] == int(Color[2]))
                    ):
                        return [i,j]
                    
                if (i == h-1 and j == 0):
                    if (
                        (copyImage[i-1,j+1][0] == int(Color[0]) and
                        copyImage[i-1,j+1][1] == int(Color[1]) and
                        copyImage[i-1,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i,j+1][0] == int(Color[0]) and
                        copyImage[i,j+1][1] == int(Color[1]) and
                        copyImage[i,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i-1,j][0] == int(Color[0]) and
                        copyImage[i-1,j][1] == int(Color[1]) and
                        copyImage[i-1,j][2] == int(Color[2]))
                    ):
                        return [i,j]
                    
                if (i == h-1 and j == w-1):
                    if (
                        (copyImage[i-1,j-1][0] == int(Color[0]) and
                        copyImage[i-1,j-1][1] == int(Color[1]) and
                        copyImage[i-1,j-1][2] == int(Color[2]))
                        or
                        (copyImage[i,j-1][0] == int(Color[0]) and
                        copyImage[i,j-1][1] == int(Color[1]) and
                        copyImage[i,j-1][2] == int(Color[2]))
                        or
                        (copyImage[i-1,j][0] == int(Color[0]) and
                        copyImage[i-1,j][1] == int(Color[1]) and
                        copyImage[i-1,j][2] == int(Color[2]))
                    ):
                        return [i,j]
                    
                if (i > 0 and i < h-1 and j == 0):
                    if (
                        (copyImage[i-1,j+1][0] == int(Color[0]) and
                        copyImage[i-1,j+1][1] == int(Color[1]) and
                        copyImage[i-1,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i,j+1][0] == int(Color[0]) and
                        copyImage[i,j+1][1] == int(Color[1]) and
                        copyImage[i,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i-1,j][0] == int(Color[0]) and
                        copyImage[i-1,j][1] == int(Color[1]) and
                        copyImage[i-1,j][2] == int(Color[2]))
                        or
                        (copyImage[i+1,j+1][0] == int(Color[0]) and
                        copyImage[i+1,j+1][1] == int(Color[1]) and
                        copyImage[i+1,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i+1,j][0] == int(Color[0]) and
                        copyImage[i+1,j][1] == int(Color[1]) and
                        copyImage[i+1,j][2] == int(Color[2]))
                    ):
                        return [i,j]
                    
                if (i > 0 and i < h-1 and j == w-1):
                    if (
                        (copyImage[i-1,j-1][0] == int(Color[0]) and
                        copyImage[i-1,j-1][1] == int(Color[1]) and
                        copyImage[i-1,j-1][2] == int(Color[2]))
                        or
                        (copyImage[i,j-1][0] == int(Color[0]) and
                        copyImage[i,j-1][1] == int(Color[1]) and
                        copyImage[i,j-1][2] == int(Color[2]))
                        or
                        (copyImage[i-1,j][0] == int(Color[0]) and
                        copyImage[i-1,j][1] == int(Color[1]) and
                        copyImage[i-1,j][2] == int(Color[2]))
                        or
                        (copyImage[i+1,j][0] == int(Color[0]) and
                        copyImage[i+1,j][1] == int(Color[1]) and
                        copyImage[i+1,j][2] == int(Color[2]))
                        or
                        (copyImage[i+1,j-1][0] == int(Color[0]) and
                        copyImage[i+1,j-1][1] == int(Color[1]) and
                        copyImage[i+1,j-1][2] == int(Color[2]))
                    ):
                        return [i,j]
                    
                if (j > 0 and j < h-1 and i == w-1):
                    if (
                        (copyImage[i-1,j-1][0] == int(Color[0]) and
                        copyImage[i-1,j-1][1] == int(Color[1]) and
                        copyImage[i-1,j-1][2] == int(Color[2]))
                        or
                        (copyImage[i,j-1][0] == int(Color[0]) and
                        copyImage[i,j-1][1] == int(Color[1]) and
                        copyImage[i,j-1][2] == int(Color[2]))
                        or
                        (copyImage[i-1,j][0] == int(Color[0]) and
                        copyImage[i-1,j][1] == int(Color[1]) and
                        copyImage[i-1,j][2] == int(Color[2]))
                        or
                        (copyImage[i,j+1][0] == int(Color[0]) and
                        copyImage[i,j+1][1] == int(Color[1]) and
                        copyImage[i,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i-1,j+1][0] == int(Color[0]) and
                        copyImage[i-1,j+1][1] == int(Color[1]) and
                        copyImage[i-1,j+1][2] == int(Color[2]))
                    ):
                        return [i,j]
        
                if (j > 0 and j < h-1 and i == 0):
                    if (
                        (copyImage[i+1,j-1][0] == int(Color[0]) and
                        copyImage[i+1,j-1][1] == int(Color[1]) and
                        copyImage[i+1,j-1][2] == int(Color[2]))
                        or
                        (copyImage[i,j-1][0] == int(Color[0]) and
                        copyImage[i,j-1][1] == int(Color[1]) and
                        copyImage[i,j-1][2] == int(Color[2])) 
                        or
                        (copyImage[i+1,j][0] == int(Color[0]) and
                        copyImage[i+1,j][1] == int(Color[1]) and
                        copyImage[i+1,j][2] == int(Color[2]))
                        or
                        (copyImage[i,j+1][0] == int(Color[0]) and
                        copyImage[i,j+1][1] == int(Color[1]) and
                        copyImage[i,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i+1,j+1][0] == int(Color[0]) and
                        copyImage[i+1,j+1][1] == int(Color[1]) and
                        copyImage[i+1,j+1][2] == int(Color[2]))
                    ):
                        return [i,j]
                    
                else:
                    if (
                        (copyImage[i+1,j][0] == int(Color[0]) and
                        copyImage[i+1,j][1] == int(Color[1]) and
                        copyImage[i+1,j][2] == int(Color[2]))
                        or
                        (copyImage[i,j-1][0] == int(Color[0]) and
                        copyImage[i,j-1][1] == int(Color[1]) and
                        copyImage[i,j-1][2] == int(Color[2])) 
                        or
                        (copyImage[i+1,j-1][0] == int(Color[0]) and
                        copyImage[i+1,j-1][1] == int(Color[1]) and
                        copyImage[i+1,j-1][2] == int(Color[2]))
                        or
                        (copyImage[i,j+1][0] == int(Color[0]) and
                        copyImage[i,j+1][1] == int(Color[1]) and
                        copyImage[i,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i+1,j+1][0] == int(Color[0]) and
                        copyImage[i+1,j+1][1] == int(Color[1]) and
                        copyImage[i+1,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i-1,j+1][0] == int(Color[0]) and
                        copyImage[i-1,j+1][1] == int(Color[1]) and
                        copyImage[i-1,j+1][2] == int(Color[2]))
                        or
                        (copyImage[i-1,j][0] == int(Color[0]) and
                        copyImage[i-1,j][1] == int(Color[1]) and
                        copyImage[i-1,j][2] == int(Color[2]))
                        or
                        (copyImage[i-1,j-1][0] == int(Color[0]) and
                        copyImage[i-1,j-1][1] == int(Color[1]) and
                        copyImage[i-1,j-1][2] == int(Color[2]))
                    ):
                        return [i,j]
                #print(copyImage[i,j])
                #cv2.imshow("copy", copyImage)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                
                #return [i,j]
                
                
    #print(tempX, tempY)
    #print(copyImage[114,78]
    #print(Color)
    #cv2.imshow("copy", copyImage)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(counter)

    

def Following_Algorithm(points, res):
    tempX = 0
    tempY = 0
    copyImage = copy.copy(res)
    tempList = []
    for i in points:
        tempX = i[0]
        tempY = i[1]
        rememberPoint = [tempX, tempY]
        flag = True
        c = [tempX, tempY]
        #if (tempX == 0 || tempX == h-1):
        #    for j in range(tempX, w):
        #        if (copyImage[j,tempY].all() == copyImage[tempX, tempY].all()):
        #            tempX = j
        #            c = [tempX, tempY]

        #if (tempX - 1 < 0):
        tempList.append(c)
        p = 0
        if (c[0] == 0  or c[0] == h-2 or c[1] == 0 or c[1] == w-2):
            new = touchborder(c, copyImage, rememberPoint)
            c = new[0]
            p = new[1]
        #print("remember",rememberPoint)
        while(flag):
            for n in range(p, 14):
                #print("p value", p)
                #print(tempList)
                if n > 7:
                    n = n - 8
                tempPoint = N_eight(n,c)
                #print("tempPoint", tempPoint, n)
                
                if ((copyImage[tempPoint[0], tempPoint[1]])[0] == (copyImage[rememberPoint[0], rememberPoint[1]])[0]
                   and (copyImage[tempPoint[0], tempPoint[1]])[1] == (copyImage[rememberPoint[0], rememberPoint[1]])[1]
                   and (copyImage[tempPoint[0], tempPoint[1]])[2] == (copyImage[rememberPoint[0], rememberPoint[1]])[2]
                   ):
                #if (np.all(copyImage[tempPoint[0], tempPoint[1]], copyImage[rememberPoint[0], rememberPoint[1]])):
                    #print("nvalue",n)
                    #ccopy = copy.deepcopy(c)
                    #print("copyhere", ccopy)
                    #print("tempPointEnter", tempPoint, n)
                    #c = N_eight(n, c)
                    c = tempPoint
                    #print("c_enter",c)
                    if (n == 0):
                        p = n
                    else:
                        p = n-1   
                    if(tempPoint != rememberPoint):
                        #if touch the border
                        if (c[0] == 0 or c[0] == h-2 or c[1] == 0 or c[1] == w-2):
                            new = touchborder(c, copyImage, rememberPoint)
                            if (new[2] == 1):
                                flag = False
                            c = new[0]
                            p = new[1]
                           # print("c,p",c,p)
                    
                    tempList.append(c)
                    #print(tempList)
                    break
            if (c[0] == rememberPoint[0] and c[1] == rememberPoint[1]):
                flag = False
        egdeList.append(tempList)
#    print(points)
#    print(tempList)

    return

def N_eight(n, c):
    if n > 7:
        n = n - 8
    i = c[0]
    j = c[1]
    if n == 0 or n == 8:
        return[i - 1, j]
    elif n == 1 or n == 9:
        return [i - 1, j + 1]
    elif n == 2 or n == 10:
        return [i, j + 1]
    elif n == 3 or n == 11:
        return [i + 1, j + 1]
    elif n ==4 or n == 12:
        return [i + 1, j]
    elif n ==5 or n == 13:
        return [i + 1, j - 1]
    elif n ==6:
        return [i, j - 1]
    elif n ==7:
        return [i - 1, j - 1]

def touchborder(c, copyImage, rememberPoint):
    if (c[0] == 0):
        #top?
        p = 2
        while ((copyImage[c[0], c[1]])[0] == (copyImage[rememberPoint[0], rememberPoint[1]])[0] and (copyImage[c[0], c[1]])[1] == (copyImage[rememberPoint[0], rememberPoint[1]])[1] and (copyImage[c[0], c[1]])[2] == (copyImage[rememberPoint[0], rememberPoint[1]])[2]):
            if (c[0] == rememberPoint[0] and c[1] == rememberPoint[1]):
                return [c,p,1]
                break
            c[1] = c[1] + 1
        c[1] = c[1] - 1
        return [c,p,0]
    elif (c[1] == 0):
        #left
        p = 0
        while ((copyImage[c[0], c[1]])[0] == (copyImage[rememberPoint[0], rememberPoint[1]])[0] and (copyImage[c[0], c[1]])[1] == (copyImage[rememberPoint[0], rememberPoint[1]])[1] and (copyImage[c[0], c[1]])[2] == (copyImage[rememberPoint[0], rememberPoint[1]])[2]):
            if (c[0] == rememberPoint[0] and c[1] == rememberPoint[1]):
                return [c,p,1]
                break
            c[0] = c[0] - 1
        c[0] = c[0] + 1
        return [c,p,0]
    elif (c[1] == w-1):
        #right
        p = 4
        while ((copyImage[c[0], c[1]])[0] == (copyImage[rememberPoint[0], rememberPoint[1]])[0] and (copyImage[c[0], c[1]])[1] == (copyImage[rememberPoint[0], rememberPoint[1]])[1] and (copyImage[c[0], c[1]])[2] == (copyImage[rememberPoint[0], rememberPoint[1]])[2]):
            if (c[0] == rememberPoint[0] and c[1] == rememberPoint[1]):
                return [c,p,1]
                break
            c[0] = c[0] + 1
        c[1] = c[1] -1
        return [c,p,0]
    elif (c[0] == h-1):
        #bottom
        p = 6
        while ((copyImage[c[0], c[1]])[0] == (copyImage[rememberPoint[0], rememberPoint[1]])[0] and (copyImage[c[0], c[1]])[1] == (copyImage[rememberPoint[0], rememberPoint[1]])[1] and (copyImage[c[0], c[1]])[2] == (copyImage[rememberPoint[0], rememberPoint[1]])[2]):
            if (c[0] == rememberPoint[0] and c[1] == rememberPoint[1]):
                return [c,p,1]
                break
            c[1] = c[1] - 1
        c[1] = c[1] + 1
        return [c,p,0]

def cutEdges(egdeList, res, BackgroudColor):
    for i in egdeList:
        for j in i:
            tempSelf = []
            tempCut = []
            selfColor = res[j[0],j[1]]
            if (j[0] > 0 and j[0] < h-1 and j[1] > 0 and j[1] < h-1):
                if (
                (
                (res[j[0],j[1]-1][0] == int(BackgroudColor[0]) or res[j[0],j[1]-1][0] == int(selfColor[0])) and
                (res[j[0],j[1]-1][1] == int(BackgroudColor[1]) or res[j[0],j[1]-1][1] == int(selfColor[1])) and
                (res[j[0],j[1]-1][2] == int(BackgroudColor[2]) or res[j[0],j[1]-1][2] == int(selfColor[2]))   
                )
                and
                (
                (res[j[0]+1,j[1]-1][0] == int(BackgroudColor[0]) or res[j[0]+1,j[1]-1][0] == int(selfColor[0])) and
                (res[j[0]+1,j[1]-1][1] == int(BackgroudColor[1]) or res[j[0]+1,j[1]-1][1] == int(selfColor[1])) and
                (res[j[0]+1,j[1]-1][2] == int(BackgroudColor[2]) or res[j[0]+1,j[1]-1][2] == int(selfColor[2]))   
                )
                and
                (
                (res[j[0]+1,j[1]][0] == int(BackgroudColor[0]) or res[j[0]+1,j[1]][0] == int(selfColor[0])) and
                (res[j[0]+1,j[1]][1] == int(BackgroudColor[1]) or res[j[0]+1,j[1]][1] == int(selfColor[1])) and
                (res[j[0]+1,j[1]][2] == int(BackgroudColor[2]) or res[j[0]+1,j[1]][2] == int(selfColor[2]))   
                )
                and
                (
                (res[j[0]+1,j[1]+1][0] == int(BackgroudColor[0]) or res[j[0]+1,j[1]+1][0] == int(selfColor[0])) and
                (res[j[0]+1,j[1]+1][1] == int(BackgroudColor[1]) or res[j[0]+1,j[1]+1][1] == int(selfColor[1])) and
                (res[j[0]+1,j[1]+1][2] == int(BackgroudColor[2]) or res[j[0]+1,j[1]+1][2] == int(selfColor[2]))   
                )
                and
                (
                (res[j[0],j[1]+1][0] == int(BackgroudColor[0]) or res[j[0],j[1]+1][0] == int(selfColor[0])) and
                (res[j[0],j[1]+1][1] == int(BackgroudColor[1]) or res[j[0],j[1]+1][1] == int(selfColor[1])) and
                (res[j[0],j[1]+1][2] == int(BackgroudColor[2]) or res[j[0],j[1]+1][2] == int(selfColor[2]))   
                )
                and
                (
                (res[j[0]-1,j[1]-1][0] == int(BackgroudColor[0]) or res[j[0]-1,j[1]-1][0] == int(selfColor[0])) and
                (res[j[0]-1,j[1]-1][1] == int(BackgroudColor[1]) or res[j[0]-1,j[1]-1][1] == int(selfColor[1])) and
                (res[j[0]-1,j[1]-1][2] == int(BackgroudColor[2]) or res[j[0]-1,j[1]-1][2] == int(selfColor[2]))   
                )
                and
                (
                (res[j[0]-1,j[1]][0] == int(BackgroudColor[0]) or res[j[0]-1,j[1]][0] == int(selfColor[0])) and
                (res[j[0]-1,j[1]][1] == int(BackgroudColor[1]) or res[j[0]-1,j[1]][1] == int(selfColor[1])) and
                (res[j[0]-1,j[1]][2] == int(BackgroudColor[2]) or res[j[0]-1,j[1]][2] == int(selfColor[2]))   
                )
                and
                (
                (res[j[0]+1,j[1]-1][0] == int(BackgroudColor[0]) or res[j[0]+1,j[1]-1][0] == int(selfColor[0])) and
                (res[j[0]+1,j[1]-1][1] == int(BackgroudColor[1]) or res[j[0]+1,j[1]-1][1] == int(selfColor[1])) and
                (res[j[0]+1,j[1]-1][2] == int(BackgroudColor[2]) or res[j[0]+1,j[1]-1][2] == int(selfColor[2]))   
                )
                ):
                    #print("there",j)
                    SelfEdges.append(j)
                else:
                    #print("here")
                    CuttingEdges.append(j)
        #SelfEdges.append(tempSelf)
        #CuttingEdges.append(tempCut)
    return

def qietu(res, contour, color):
    img = copy.deepcopy(res)
    for i in range(len(contour)):
        img[contour[i][0]][contour[i][1]] = color
        
    cv2.imshow(str(color), img)
    
    return

def drawLine(res):
    copyImg = copy.deepcopy(res)
    for i in range(h):
        for j in range(w):
                copyImg[i,j][0] = 0
                copyImg[i,j][1] = 0
                copyImg[i,j][2] = 0
    for i in SelfEdges:
        copyImg[i[0],i[1]][0] = 255
        copyImg[i[0],i[1]][1] = 0
        copyImg[i[0],i[1]][2] = 0
    for i in BackgroundEgdes:
        copyImg[i[0],i[1]][0] = 0
        copyImg[i[0],i[1]][1] = 255
        copyImg[i[0],i[1]][2] = 0
    for i in CuttingEdges:
        copyImg[i[0],i[1]][0] = 0
        copyImg[i[0],i[1]][1] = 0
        copyImg[i[0],i[1]][2] = 255
    cv2.imshow("border", copyImg)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
    return

def changeColor(res,colorList):
   # print("color",colorList)
    
    temp = []
    for x in range(len(colorList[:-1])):
        #print(x)
        colorList[x] = [int(i) for i in colorList[x]]
        img = copy.deepcopy(res)
        for i in range(h):
            
            for j in range(w):
               # print(res[i][j])
               # print(res[i][j])
               # print(colorList[-1])
                
                if (res[i][j][0] == colorList[x][0] and res[i][j][1] == colorList[x][1] and res[i][j][2] == colorList[x][2]):
                   # print("hello")
                    img[i][j] = [255,255,255]
                else:
                    img[i][j] = [0,0,0]       
        temp.append(img)
        
    return temp



def shoot(contours):
    start = contours[0]
#    print(start)
    contours = np.array(contours)
    err_sq = 0
    
  #  err_you = 0
  #  err_youxia = 0
    
    
    max_j = max(contours, key=lambda item: item[1])
    
 #   max_i = contours.max(0)
    
   # print(max_j[0][1])
   # print(start)
    #print(max_j)
   # print(start[1])
   
   
    for i in range(max_j[1]-start[1]):
      #  print(contours[i])
      # print(start[0])
      #  print(start[1]+i)
        err_sq += dist(contours[i], [start[0], start[1]+i])
    
    err_tri = 0
    
    for i in range(max_j[1]-start[1]):
        #print("contours, ", contours[i])
        #print("line, ", [start[0]+math.sqrt(3)*i, start[1]+i])
        err_tri += dist(contours[i], [start[0]+ 0.577 * i, start[1]+0.8*i])
    
    print("square error", err_sq)
    print("triangle error", err_tri)
    print("So the error ratio is ", err_tri/err_sq)
    
        
    if ((err_tri/err_sq>0.333) &  (err_tri/err_sq<3.33)):
       # print("circle")
        return [start,"circle"]
    elif (err_tri/err_sq>3.33):
      #  print("square")
        return [start,"square"]
    elif (err_tri/err_sq<0.33):
     #   print("triangle")
        return [start,"triangle"]
    
    else:
       # print("Unknown error")
        return [start,"fail"]

def dist(a,b):
    return (b[0] - a[0])**2 + (b[1]-a[1])**2

dst = denoising(img)

a = countPixel(img)

counter = 0

#can be improved
for i in range (1,6):
    if (a[-i][1] >= 220):
        counter += 1

res = kmeans(dst, counter)

c = countPixel(res)

#can be improved
for i in range(len(c)):
    if c[i][1] <= 560:
        counter -= 1

res = kmeans(dst, counter)

c = countPixel(res)

ColorList = []
for i in c:
    temp = i[0]
    temp = temp.replace("[", "")
    temp = temp.replace("]", "")
    temp = temp.replace(" ", "")
    temp = temp.split(",")
    ColorList.append(temp)
        

points = []        

x = changeColor(res,ColorList)

ColorList = ColorList[:-1]

for i in ColorList:
    points.append(Binarize(i,res))
#print(points)
Following_Algorithm(points, res)

BackgroudColor = countPixel(res)[-1][0]

BackgroudColor = BackgroudColor.replace("[", "")
BackgroudColor = BackgroudColor.replace("]", "")
BackgroudColor = BackgroudColor.replace(" ", "")
BackgroudColor = BackgroudColor.split(",")



#findBackgroundContour(img, res, BackgroudColor)
#print(egdeList)
cutEdges(egdeList, res, BackgroudColor)
#print("This is self ", len(SelfEdges))
#print("This is cutting ", len(CuttingEdges))
#print(len(egdeList))
#backedges = copy.deepcopy(egdeList)

findBackgroundContour(img, res, BackgroudColor)
#print("This is self ", len(SelfEdges))
#print("This is cutting ", len(CuttingEdges))
#print("This is backgroundContour ", BackgroundEgdes)

#qietu(res, BackgroundEgdes, [0,255,0])
#qietu(res, SelfEdges, [255,0,0])
#ßqietu(res, CuttingEdges, [0,0,255])
#drawLine(res)


for i in range(len(egdeList)):
    pos, shape_ = shoot(egdeList[i])
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos.reverse()
    pos = tuple(pos)
   # print(pos)
    print("So it's a " + shape_)
    cv2.putText(res, shape_, pos, font, 0.3, (0,0,0))




#for i in range(len(x)):
#    cv2.imshow(str(i), x[i])

#cv2.imshow("ori", img)
#cv2.imshow("aft", dst)
cv2.imshow("res", res)
#cv2.imwrite("1211.png", res)
cv2.waitKey(0)
cv2.destroyAllWindows()