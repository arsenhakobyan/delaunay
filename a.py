#!/usr/bin/env python
import numpy as np
import cv2
import sys
import random
 

fg_pointsNumber = 750
bg_pointsNumber = 60
GrabIterations = 15 # this is yet an experimental value, but probably should be updated to be dependent from an image features.
fg_d = 3
bg_d = 6
fg_blur = 3;
bg_blur = 9;


def show(name, img, val = 0):
    cv2.imshow(name, img)
    cv2.waitKey(val)

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
 
# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0 )
 
 
# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        #if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            #cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            #cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            #cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
 
 
# Draw voronoi diagram
def draw_voronoi(img, subdiv, contours, alpha):
    src = img.copy()
    areamask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(areamask, contours, (255, 255, 255))
    ( facets, centers) = subdiv.getVoronoiFacetList([])

 
    for i in xrange(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
         
        ifacet = np.array(ifacet_arr, np.int)
        newmask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(newmask, np.int32([ifacet]), (255, 255, 255))

        ff = cv2.bitwise_and(newmask, areamask)
        ff = 255*(ff.astype('uint8'))
        im, contours, hierarchy = cv2.findContours(ff,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if (len(contours) <= 0):
            continue
        cnt = contours[0]
        if (0 >= cv2.contourArea(cnt)):
            continue
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        col = src[cY][cX]
        color = ((int(col[0]), int(col[1]), int(col[2])))

        #color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
 
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        #cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        #ifacets = np.array([ifacet])
        #cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        #cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)
    cv2.addWeighted(img, alpha, src, 1 - alpha, 0, img)
 
def cutContour(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (2,2,img.shape[1],img.shape[0])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, GrabIterations, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    newimg = img * mask2[:,:,np.newaxis]
    return (newimg, mask2)

def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
	random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
	if (random_point.within(poly)):
	    points.append(random_point)

    return points

def generate_delaunay_contours(img, contours, points, alpha):
    # Insert points into subdiv
    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])
     
    subdiv = cv2.Subdiv2D(rect);
    for p in points :
        subdiv.insert(p)
         
        # Show animation
        if animate :
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay( img_copy, subdiv, (255, 255, 255) );
            #cv2.imshow(win_delaunay, img_copy)
            #cv2.waitKey(100)
 
    # Draw delaunay triangles
    draw_delaunay( img, subdiv, (255, 255, 255) );
 
    ## Draw points
    #for p in points :
    #    draw_point(img, p, (0,0,255))
 
    # Allocate space for Voronoi Diagram
    #img_voronoi = np.zeros(img.shape, dtype = img.dtype)
    img_voronoi = img.copy()
 
    # Draw Voronoi diagram
    draw_voronoi(img_voronoi, subdiv, contours, alpha)
 
    # Show results
    #cv2.imshow(win_delaunay,img)
    #cv2.imshow(win_voronoi,img_voronoi)
    return img_voronoi

def getPointsInContour(contours, pointsNumber, ROI):
    points = []
    x, y, w, h = ROI
    i = 0
    while (i < pointsNumber):
        p = (random.randint(x, x + w), random.randint(y, y + h))
        for cnt in contours:
            if (cv2.pointPolygonTest(cnt, p, False) >= 0):
                points.append(p)
                i += 1
    return points

if __name__ == '__main__':
 
    # Define window names
    win_delaunay = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"
 
    # Turn on animation while drawing triangles
    animate = False
     
    # Define colors for drawing.
    delaunay_color = (255,255,255)
    points_color = (0, 0, 255)
 
    # Read in the image.
    img = cv2.imread(sys.argv[1])
    #cv2.imshow("input_Image", img)

    # Foregraund and Background separation.
    foreground, fg_mask = cutContour(img)
    fg = foreground.copy()
    bg = foreground.copy()
    background = cv2.subtract(img, foreground) 
    fg_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    fg_im2, fg_contours, fg_hierarchy = cv2.findContours(fg_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bg_im2, bg_contours, bg_hierarchy = cv2.findContours(bg_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fg, fg_contours, -1, (0, 255, 222), 1)
    cv2.drawContours(bg, bg_contours, -1, (255, 0, 222), 1)

    fg_points = getPointsInContour(fg_contours, fg_pointsNumber, (1, 1, img.shape[1] - 2, img.shape[0] - 2))
    bg_points = getPointsInContour(bg_contours, bg_pointsNumber, (1, 1, img.shape[1] - 2, img.shape[0] - 2))

     # Keep a copy around
    img_orig = img.copy();
     
    fg_voronoi = generate_delaunay_contours(foreground, fg_contours, fg_points, 0.5)
    bg_voronoi = generate_delaunay_contours(background, bg_contours, bg_points, 0.5)

    fimg = img.copy()
    bimg = img.copy()

    fimg = cv2.bitwise_or(fg_voronoi, fg_voronoi, mask = fg_mask)
    #fimg = cv2.GaussianBlur(fimg,(fg_blur,fg_blur),0)
    #fimg = cv2.medianBlur(fimg, fg_blur)
    fimg = cv2.bilateralFilter(fimg, fg_d, fg_blur, fg_blur)
    show("fimg", fimg)

    bg_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(bg_mask, bg_contours, (255, 255, 255))

    bimg = cv2.bitwise_or(bg_voronoi, bg_voronoi, mask = bg_mask)
    #bimg = cv2.GaussianBlur(bimg,(bg_blur,bg_blur),0)
    #bimg = cv2.medianBlur(bimg, bg_blur)
    bimg = cv2.bilateralFilter(bimg, bg_d, bg_blur, bg_blur)
    show("bimg", bimg)

    output = cv2.bitwise_or(fimg, bimg)
    cv2.imwrite("output.jpg", output)
    show("output", output)



