# Camille Marie H. Tatoy
# 2015-11050
# CoE197M-THY

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import scipy
from scipy import linalg, matrix
from scipy.interpolate import  griddata

def image2array(im):  
    try:
        imx = Image.open(im)    # used PIL to transform image
        return np.array(imx)    # transformed into an array
    except IOError:
        return []

# --------------------------------------------------

def select_pts(im):       
     plt.imshow(im)             # show image
     xx = plt.ginput(4)         # use pyplot to select four points
     xx.append(xx[0])           # saved into vector of coordinates of points selected
                                # append first pt at end to form ractangle
     return xx

# --------------------------------------------------

# Generate ground truth
def get_rect(xx):
    plt.figure(2)
    
    zzd = np.zeros((5,3))   # rows, columns
    for ii in range(len(xx)-1):         
        x1 = xx[ii][0]; y1 = xx[ii][1]
        zzd[ii,0] = x1; zzd[ii,1] = y1; zzd[ii,2] = 1; 
        plt.plot([xx[ii][0],xx[ii+1][0]], [xx[ii][1],xx[ii+1][1]], 'ro-')   # show 4 pts selected

    jj = 0
    aa = [0,0,1,0,1,3,0,3]                                                  # (0,1) --- (3,1) 
                                                                            #   |         |
                                                                            # (0,0) --- (3,0)
    zz = np.zeros((5,3))     
    for ii in range(len(zzd)-1):
            zz[ii,0] = zzd[aa[jj],0] 
            zz[ii,1] = zzd[aa[jj+1],1] 
            zz[ii,2] = 1;   
            jj = jj+2
    zz[4,:] = zz[0,:]                                                       # [initital:end:indexjump]

    # stackoverflow.com/questions/3190483/transform-quadrilateral-into-a-rectangle
    # stackoverflow.com/questions/33283088/transform-irregular-quadrilateral-to-rectangle-in-python-matplotlib

    for ii in range(4):      
        plt.plot([zz[ii,0],zz[ii+1,0]], [zz[ii,1],zz[ii+1,1]], 'go-')       # show adjusted rectangle

    plt.show()
    return zz[0:4,:],zzd[0:4,:]    
        
# --------------------------------------------------

def norm(zz):
    # Translate coordinates to center
    uu = zz.T
    ff_xx = np.ones(uu.shape)
    indices, = np.where(abs(uu[2,:]) > 10**-12)
    ff_xx[0:2,indices] = uu[0:2,indices]/uu[2,indices]
    ff_xx[2,indices]  = 1.

    mu = np.mean(ff_xx[0:2,:],axis = 1)                                 # Obtain center of the region by taking mean of the points
    mu_r = np.zeros((mu.shape[0],ff_xx.shape[1]))                       # Extend to a vector
    
    for ii in range(ff_xx.shape[1]):
        mu_r[:,ii] = mu
                                                                        # subtract mean from the points
    mu_dist = np.mean((np.sum((ff_xx[0:2] - mu_r)**2,axis =0))**0.5)    # average of Euclidean distance between the points and the center of the region

    # Scale such that average distance x to center is √2
    scale =  (2**0.5/mu_dist)

    #Translation - move the center by scale value
    s0 = -scale*mu[0]
    s1 = -scale*mu[1]

    # Scaling Matrix, S          ( s  0  s0 )
    #                            | 0  s  s1 |
    #                            ( 0  0   1 )
    S = np.array([[scale, 0, s0],[0, scale, s1], [0, 0, 1]])
    normalized_zz = S@ff_xx                                             # Product of transformation and the points
    return normalized_zz, S

# --------------------------------------------------

def compute_A(uu,vv):
    # uu = GT = x' in HZ         --> points in rectangle
    # vv = distorted = x in HZ   --> points in original
    A = np.zeros((2*(uu.shape[0]+1),9))
    jj = 0
    for ii in range(uu.shape[0]+1):
        # Compute for coefficients
        # last corrdinate of rectangle time ith pt in original image
        a = (  np.zeros((1,3))[0]  )                                    # for the two zeroes
        b = ( -uu[2,ii] * vv[:,ii] ) 
        c =    uu[1,ii] * vv[:,ii]
        d =    uu[2,ii] * vv[:,ii]
        f = ( -uu[0,ii] * vv[:,ii] )

        # Concatenate in first row and second row
        row1 = np.concatenate((a, b, c), axis=None)
        row2 = np.concatenate((d, a, f), axis=None)
        A[jj,:] = row1
        A[jj+1,:] = row2
        jj = jj+2

        # Repeat for all A_i matrices, i=1:4
    return A

# --------------------------------------------------

# Compute 1-D null-space of matrix A
# Ah=0
def compute_H(A,T1,T2):
    null_space_of_A = -scipy.linalg.null_space(A)
    hh_normalized = np.reshape(null_space_of_A,(3,3)) 
    hh = np.dot(np.linalg.inv(T2),np.dot(hh_normalized,T1))
    return hh

# --------------------------------------------------

# Transform Image
def image_rebound(mm,nn,hh):
    W = np.array([[1, nn, nn, 1 ],[1, 1, mm, mm],[ 1, 1, 1, 1]])
    ws = np.dot(hh,W)
    # Scaling
    xx = np.vstack((ws[2,:],ws[2,:],ws[2,:]))
    wsX =  np.round(ws/xx)
    bounds = [np.min(wsX[1,:]), np.max(wsX[1,:]),np.min(wsX[0,:]), np.max(wsX[0,:])]
    return bounds

# --------------------------------------------------

# Build transform for the bounds
def transform(imm,hh):   
    mm,nn = imm.shape[0],imm.shape[0]
    bounds = image_rebound(mm,nn,hh)
    nrows = bounds[1] - bounds[0]
    ncols = bounds[3] - bounds[2]
    s = max(nn,mm)/max(nrows,ncols)
    scale = np.array([[s, 0, 0],[0, s, 0], [0, 0, 1]])
    trasf = scale@hh
    trasf_prec =  np.linalg.inv(trasf)
    bounds = image_rebound(mm,nn,trasf)
    nrows = (bounds[1] - bounds[0]).astype(np.int)
    ncols = (bounds[3] - bounds[2]).astype(np.int)
    return bounds, nrows, ncols, trasf, trasf_prec

# --------------------------------------------------

def array2image(nrows,ncols,im,bounds,trasf_prec,nsamples):
    xx  = np.linspace(1, ncols, ncols)
    yy  = np.linspace(1, nrows, nrows)
    [xi,yi] = np.meshgrid(xx,yy) 

    # Reshape original image
    a0 = np.reshape(xi, -1,order ='F')+bounds[2]
    a1 = np.reshape(yi,-1, order ='F')+bounds[0]
    a2 = np.ones((ncols*nrows))
    uv = np.vstack((a0.T,a1.T,a2.T)) 
    new_trasf = np.dot(trasf_prec,uv)
    val_normalization = np.vstack((new_trasf[2,:],new_trasf[2,:],new_trasf[2,:]))
   
    # The new transformation
    newT = new_trasf/val_normalization
    
    # Resample points from original image
    xi = np.reshape(newT[0,:],(nrows,ncols),order ='F') 
    yi = np.reshape(newT[1,:],(nrows,ncols),order ='F')
    cols = im.shape[1]
    rows = im.shape[0]
    xxq  = np.linspace(1, rows, rows).astype(np.int)
    yyq  = np.linspace(1, cols, cols).astype(np.int)
    [x,y] = np.meshgrid(yyq,xxq) 
    x = (x - 1).astype(np.int) # Offset x and y relative to region origin.
    y = (y - 1).astype(np.int) 
        
    ix = np.random.randint(im.shape[1], size=nsamples)
    iy = np.random.randint(im.shape[0], size=nsamples)
    samples = im[iy,ix]
    
    # Interpolate the points of the original image via samples into coordinates of new image
    int_im = griddata((iy,ix), samples, (yi,xi))
    
    plt.imshow(int_im.astype(np.uint8))
    plt.show()