Camille Marie H. Tatoy
2015-11050
CoE197M-THY


Machine Problem 1:
Remove projective distortion on a given image
at least 4pts on target image known

Theorem:
A mapping h: P^2 -> P^2 is a projectivity iff there exists a non-singular 3x3 matrix H 
such that for any point in P^2 represented by a vector x it is true that h(x) = Hx
    
Projective Transform
A planar projective transformation is a linear transformation on homogeneous 3-vectors 
represented by a non-singular 3x3 matrix:

( x'_1 )    [ h_11 h_12 h_13 ] ( x_1 )
| x'_2 | =  | h_21 h_22 h_23 | | x_2 |
( x'_3 )    [ h_31 h_32 h_33 ] ( x_3 ) 

H is a homogeneous matrix and it has 8 degrees of freedom (DOF)


Select four points corresponding to a planar section 
such that rectangle if no projective distortion
   x4 <----- x3                  x4'<----- x3'
  /         /                     |         |
 /         /                      |         |
x1 -----> x2                     x1'-----> x2'