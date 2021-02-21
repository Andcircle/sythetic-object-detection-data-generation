import argparse
import glob
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import pi
import random


def _parse_args():
    parser = argparse.ArgumentParser(description='This script is used to do traffic sign augmentation'
                                    '(including roll, yaw, pitch, scale, color tune, color noize)', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_path',
                        help='input path to test image',
                        default='./image_argb/sign80.png')

    return parser.parse_args()


def get_transform_matrix(image, theta, phi, gamma, dx, dy, dz, f, scale):
    """ Get Perspective Projection Matrix """
        
    h, w, _=image.shape

    # Projection 2D -> 3D matrix, move orginal points to center of image
    M = np.array([ [1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 0],
                    [0, 0, 1]])
    
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])
    
    RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
    
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix, dz is set to focal length, so projection is 1:1
    T = np.array([  [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    P = np.array([ [f, 0, 0, 0],
                    [0, f, 0, 0],
                    [0, 0, 1, 0]])

    # Scale the transform and then move origin back to upper left corner
    S = np.array([ [scale, 0, w/2],
                    [0, scale, h/2],
                    [0, 0, 1]]) 

    # Final transformation matrix
    return np.dot(S, np.dot(P, np.dot(T, np.dot(R, M))))


def transform_image(image, theta=0., phi=0., gamma=0., dx=0., dy=0., dz=0., scale=1.):
    """
    theta rotation around x-axis
    phi rotation around y-axis
    gamma rotation around z-axis
    dx shift along x-axis
    dy shift along y-axis
    dz shift along z-axis
    scale
    """
        
    # Get radius of rotation along 3 axes
    rtheta = theta / 180 * pi
    rphi = phi / 180 * pi
    rgamma = gamma / 180 * pi

    height, width, _ = image.shape
    
    # Get ideal focal length on z axis
    # NOTE: Change this section to other axis if needed
    d = np.sqrt(height**2 + width**2)
    focal = d

    # Get projection matrix
    mat = get_transform_matrix(image, rtheta, rphi, rgamma, dx, dy, focal, focal, scale)

    # Adjust to new traffic sign bound
    p = np.array([[0, 0, 1],
                  [width-1, 0, 1],
                  [width-1, height-1, 1],
                  [0, height-1, 1]])
    p = np.dot(p, np.transpose(mat))
    p = p/np.expand_dims(p[:,2],-1)
    p = p.astype('int')
    xmin=min(p[:,0])
    xmax=max(p[:,0])
    ymin=min(p[:,1])
    ymax=max(p[:,1])

    nwidth=xmax-xmin
    nheight=ymax-ymin

    C = np.array([ [1, 0, -xmin],
                    [0, 1, -ymin],
                    [0, 0, 1]])
    mat = np.dot(C, mat)

    result=cv2.warpPerspective(image.copy(), mat, (nwidth, nheight))

    ## Visualize affined bounding box for debug purpose
    # p = np.dot(p, np.transpose(C))[:,:2]
    # draw_contour(result, p)
    
    return result


def draw_contour(image, contour): 
    contour=contour.astype('int')

    for p1, p2 in zip(contour, contour[1:]):
        cv2.line(image, tuple(p1), tuple(p2), (0,255,0,255), 2)
    
    cv2.line(image, tuple(contour[-1]), tuple(contour[0]), (0,255,0,255), 2)


def draw_bbox(image, bbox):
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    p1=(x1, y1)
    p2=(x2, y1)
    p3=(x2, y2)
    p4=(x1, y2)

    cv2.line(image, tuple(p1), tuple(p2), (0,255,0,255), 2)
    cv2.line(image, tuple(p2), tuple(p3), (0,255,0,255), 2)
    cv2.line(image, tuple(p3), tuple(p4), (0,255,0,255), 2)
    cv2.line(image, tuple(p4), tuple(p1), (0,255,0,255), 2)


def crop_image(image):
    h, w, _ = image.shape

    xmin=w-1
    xmax=0
    ymin=h-1
    ymax=0

    for r in range(h):
        for c in range(w):
            if image[r,c,3]>=250:
                xmin=min(xmin,c)
                xmax=max(xmax,c)
                ymin=min(ymin,r)
                ymax=max(ymax,r)

    # box=np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    # draw_contour(image, box)

    return image.copy()[ymin:ymax+1, xmin:xmax+1]



def random_transform_image(image, min_scale=0.3, max_scale=0.7, max_rotate_XY=90, max_rotate_Z=45):

    theta=random.random() * max_rotate_XY - max_rotate_XY/2
    phi=random.random() * max_rotate_XY - max_rotate_XY/2
    gamma=random.random() * max_rotate_Z - max_rotate_Z/2

    scale = random.random() * (max_scale - min_scale) + min_scale

    # print("theta: {}; phi: {}; gamma:{} scale: {}".format(theta, phi, gamma, scale))

    result = transform_image(image, theta=theta, phi=phi, gamma=gamma, scale=scale)

    return crop_image(result)


def transform_test(input_path):
    
    image=cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    print("test1")
    result=transform_image(image, theta=45, phi=0, gamma=0, scale=1)
    result=crop_image(result)    
    plt.imshow(result)
    plt.show()

    print("test2")
    result=transform_image(image, theta=-45, phi=0, gamma=0, scale=1)      
    result=crop_image(result)
    plt.imshow(result)
    plt.show()

    print("test3")
    result=transform_image(image, theta=0, phi=45, gamma=0, scale=1)       
    result=crop_image(result)
    plt.imshow(result)
    plt.show()

    print("test4")
    result=transform_image(image, theta=0, phi=-45, gamma=0, scale=1)   
    result=crop_image(result)   
    plt.imshow(result)
    plt.show()

    print("test5")
    result=transform_image(image, theta=0, phi=0, gamma=45, scale=1)  
    result=crop_image(result)     
    plt.imshow(result)
    plt.show()

    print("test6")
    result=transform_image(image, theta=0, phi=0, gamma=-45, scale=1)  
    result=crop_image(result)     
    plt.imshow(result)
    plt.show()

    print("test7")
    result=transform_image(image, theta=45, phi=45, gamma=0, scale=1)  
    result=crop_image(result)    
    plt.imshow(result)
    plt.show()

    print("test8")
    result=transform_image(image, theta=45, phi=0, gamma=45, scale=1) 
    result=crop_image(result)      
    plt.imshow(result)
    plt.show()

    print("test9")
    result=transform_image(image, theta=0, phi=45, gamma=45, scale=1)
    result=crop_image(result)       
    plt.imshow(result)
    plt.show()

    print("test10")
    result=transform_image(image, theta=0, phi=0, gamma=0, scale=1)    
    result=crop_image(result)  
    plt.imshow(result)
    plt.show()

    print("test1")
    result=transform_image(image, theta=45, phi=0, gamma=45, scale=0.5)   
    result=crop_image(result)     
    plt.imshow(result)
    plt.show()

    print("test1")
    result=transform_image(image, theta=0, phi=45, gamma=45, scale=0.5)  
    result=crop_image(result)     
    plt.imshow(result)
    plt.show()

    print("test11")
    result=transform_image(image, theta=0, phi=0, gamma=0, scale=0.5)   
    result=crop_image(result)    
    plt.imshow(result)
    plt.show()

    print("test12")
    for i in range(5):
        result=random_transform_image(image)  
        plt.imshow(result)
        plt.show()

        # cv2.imwrite('./test/result_{}.png'.format(i), result)


def main():
    args = _parse_args()
    transform_test(args.input_path)

if __name__=='__main__':
    main()