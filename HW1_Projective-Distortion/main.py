# Camille Marie H. Tatoy
# 2015-11050
# CoE197M-THY

import os
from scipy import linalg, matrix
from scipy.interpolate import griddata
from functions import *
import matplotlib
matplotlib.use("TkAgg")

# Homework 1:
# Remove projective distortion on a given image
# at least 4pts on target image known

# Theorem:
# A mapping h: P^2 -> P^2 is a projectivity iff there exists a non-singular 3x3 matrix H 
# such that for any point in P^2 represented by a vector x it is true that h(x) = Hx
    
# Projective Transform
# A planar projective transformation is a linear transformation on homogeneous 3-vectors 
# represented by a non-singular 3x3 matrix:

# ( x'_1 )    [ h_11 h_12 h_13 ] ( x_1 )
# | x'_2 | =  | h_21 h_22 h_23 | | x_2 |
# ( x'_3 )    [ h_31 h_32 h_33 ] ( x_3 ) 

# H is a homogeneous matrix and it has 8 degrees of freedom (DOF)

# --------------------------------------------------

cwd = os.getcwd()
ddir = os.listdir(cwd)

image_name = "campus.jpg" 
folder_name = "pictures"
    
im = image2array(os.path.join(cwd,folder_name,image_name))                           # Load image and turn into array

# Select four points corresponding to a planar section 
# such that rectangle if no projective distortion
#    x4 <----- x3                  x4'<----- x3'
#   /         /                     |         |
#  /         /                      |         |
# x1 -----> x2                     x1'-----> x2'
pts = select_pts(im) 

# Generate rectangular figure
pts_rect, pts_dist = get_rect(pts)

# Apply normalization to both distorted and rectangular vectors            
norm_pts_dist, T1_norm = norm(pts_dist) # points in distorted image
norm_pts_rect, T2_norm = norm(pts_rect) # points in rectangular image

# Find matrix A
# A_i*h=0
A = compute_A(norm_pts_rect,norm_pts_dist)

# Find transformation H
H = compute_H(A,T1_norm,T2_norm)

# Remapping
# 1,1,1 ----- n,1,1
#   |           |
#   |           |
# 1,m,1 ----- n,m,1

# Apply transformation H to all points in the image to obtain the undistorted image        
bounds, nrows, ncols,  trasf, trasf_inv = transform(im,H)  

# Number of samples
l,w  = im.shape[0],im.shape[1]
if max(l,w)>1000:
    nsamples = 10**6 
else: 
    nsamples = 10**5  

array2image(nrows,ncols,im,bounds,trasf_inv,nsamples)

# --------------------------------------------------










# youtu.be/cJUniJ3C7I4
# geeksforgeeks.org/python-opencv-affine-transformation
# engineering.purdue.edu/kak/computervision/ECE661.08/solution/hw2_s2.pdf