import cv2
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Function Definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Cubic function
def Find_Value(a, b, c, d,x):
    return a * x**3 + b * x**2 + c * x + d

#Makes spline coefficient matrix for a given size
def Make_Spline(num_points):
    points =  np.arrange(num_points)#The number of points to be included in the spline

    '''
    ~~~~~~~~~~ Create system of equations ~~~~~~~~~~

        The goal here is to create a 3rd order function connecting each
        adjacent pair of points. We will have to perform some math to
        create a matrix consisting of the coefficients for the following
        equations:
        - polynomial: a*x^3 + b*x^2 + c*x + d = y
        - first derivative: f'_(n-1) = f'_n
        - second derivate: f"_(n-1) = f"_n
        - boundary conditions: f"_0 = f"_n = 0
        This boundary condition corresponds to a 'natural spline'
    '''

    p_count = 0 #Used for indexing points

    matrix = np.zeros(shape=(4*(num_points-1),(4*(num_points-1))), dtype=int) #Create empty square matrix

    #Fill in the 3rd order equations based on: a*x^3 + b*x^2 + c*x + d = y
    x = points[p_count] #Store the first point before loop

    for row in range(1,2*num_points-2,2):
        p_count += 1
        matrix[row-1,4*(p_count-1):4*(p_count)] = [x**3,x**2,x,1] #Use old x value
        x = points[p_count]
        matrix[row,4*(p_count-1):4*(p_count)] = [x**3,x**2,x,1] #Use new x value

    #Fill in the 1st order derivatives: f'_(n-1) = f'_n
    p_count = 0 #Reset counter

    for row in range(2*num_points-2,3*num_points-4):
        p_count += 1 #The first point is not included
        x = points[p_count]

        matrix[row,4*(p_count-1):4*(p_count)] = [3*x**2,2*x,1,0]
        matrix[row,4*(p_count):4*(p_count+1)] = -matrix[row,4*(p_count-1):4*(p_count)]
    #Fill in the 2nd order derivatives: f"_(n-1) = f"_n
    p_count = 0 #Reset counter

    for row in range(3*num_points-4,4*num_points-6):
        p_count += 1 #The first point is not included
        x = points[p_count]

        matrix[row,4*(p_count-1):4*(p_count)] = [6*x,2,0,0]
        matrix[row,4*(p_count):4*(p_count+1)] = -matrix[row,4*(p_count-1):4*(p_count)]

    #Fill in the 2nd order derivative for the boundary f"_0 = 0
    x = points[0]
    matrix[4*(num_points-1)-2,:4] = [6*x,2,0,0]
    
    #Fill in the 2nd order derivative for the boundary f"_0 = 0
    x = points[num_points-1]
    matrix[4*(num_points-1)-1,4*(num_points-2):] = [6*x,2,0,0]

    return matrix

#Returns the coefficients for a cubic spline solution
def Find_Spline(z_values,spline_mat):
    num_points = len(z_values)
    z = np.zeros(shape=(4*(num_points-1),1), dtype=float) #Column vector for the Y values of the points

    p_count = 0
    z[0] = z_values[0] #Assign first Y value

    #Each function must connect to the next so Y values are repeated
    for i in range(1,2*(num_points-1)-1,2):
        p_count += 1
        z[i] = z_values[p_count]
        z[i+1] = z_values[p_count]

        z[2*(num_points-1)-1] = z_values[num_points-1] #Assign last Y value
    
    #This solves the matrix using inverse method
    return np.linalg.inv(spline_mat).dot(z)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Function Definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

orig_img = cv2.imread("sample_image.png") #Read in image into a numpy array
orig_img_yuv = cv2.cvtColor(orig_img,cv2.COLOR_BGR2YUV) #Originally BGR, convert to YUV

#Returns the length of each dimension (X pixels, Y pixels, channels)
x_pixels, y_pixels, channels = orig_img_yuv.shape

#Grabs each channel from the full colour image as a separate array
luma_img = orig_img_yuv[:,:,0] #luma channel (Y), this is the brightness
blue_img = orig_img_yuv[:,:,1] #blue channel (U), blue projection
red_img = orig_img_yuv[:,:,2] #red channel (V), red projection

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Display orginal image and channels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cv2.imshow('Full Colour Image',orig_img) #Display full colour image (needs a conversion)

#Show all three channels separated
separated_channels = np.concatenate((luma_img,blue_img,red_img),axis=1)
#cv2.imshow('Y-channel, U-channel, V-channel',separated_channels)

original_size = x_pixels * y_pixels * channels * 8
print("Original image size: " + str(original_size) + " bits")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compress Y channel by factor of 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
luma_shrink = 2 #Factor by which to compress the image
luma_img_comp = np.zeros((x_pixels//luma_shrink,y_pixels//luma_shrink),dtype=np.uint8) #Creates empty array half the size of the original image

for x in range(0,x_pixels-1,luma_shrink):
    for y in range(0,y_pixels-1,luma_shrink):
        luma_img_comp[x//luma_shrink,y//luma_shrink] = luma_img[x,y]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compress U channel by factor of X ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
blue_shrink = 2
blue_img_comp = np.zeros((x_pixels//blue_shrink,y_pixels//blue_shrink),dtype=np.uint8) #Creates empty array half the size of the original image

for x in range(0,x_pixels-1,blue_shrink):
    for y in range(0,y_pixels-1,blue_shrink):
        blue_img_comp[x//blue_shrink,y//blue_shrink] = blue_img[x,y]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compress V channel by factor of X ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
red_shrink = 2
red_img_comp = np.zeros((x_pixels//red_shrink,y_pixels//red_shrink),dtype=np.uint8) #Creates empty array half the size of the original image

for x in range(0,x_pixels-1,red_shrink):
    for y in range(0,y_pixels-1,red_shrink):
        red_img_comp[x//red_shrink,y//red_shrink] = red_img[x,y]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Display the compressed channels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#cv2.imshow("Luma (Y) Channel Shrunk by Factor of "+str(luma_shrink),luma_img_comp)
#cv2.imshow("Blue (U) Channel Shrunk by Factor of "+str(blue_shrink),blue_img_comp)
#cv2.imshow("Red (U) Channel Shrunk by Factor of "+str(red_shrink),red_img_comp)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reconstruct the Y channel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_pixels_comp, y_pixels_comp = luma_img_comp.shape #size of the compressed Y channel

comp_factor = x_pixels // x_pixels_comp #factor by which the image was compressed

luma_img_up = np.zeros((x_pixels,y_pixels),dtype=np.uint8) #Empty array for the reconstruction

#Loop through and add all the sampled pixels from the compressed image
for x in range(0,x_pixels_comp):
    for y in range(0,y_pixels_comp):
        luma_img_up[x*comp_factor,y*comp_factor] = luma_img_comp[x,y]

#Perform a spline calculation across each row and column, averaging the centre pixels
spline_matrix = Make_Spline(4) #precalculate for the spline method, will be using 4 pixels x 4 pixels

for x in range(0,x_pixels/100,4*comp_factor):
    for y in range(0,y_pixels/100,4*comp_factor):
        spline = [] #Empty list
        spline.append(Find_Spline([luma_img_up[x,y],luma_img_up[x+1,y],luma_img_up[x+2,y],luma_img_up[x+3,y]],spline_matrix))




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reconstruct the full image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#cv2.imshow("Luma Reconstruct",luma_img_up)
#cv2.imshow("Blue Reconstruct",blue_img_up)
#cv2.imshow("Red Reconstruct",red_img_up)

#reconstructed_img = np.zeros((x_pixels,y_pixels,3),dtype=np.uint8)

#reconstructed_img[:,:,0] = luma_img_up
#reconstructed_img[:,:,1] = blue_img_up
#reconstructed_img[:,:,2] = red_img_up

#reconstructed_img = cv2.cvtColor(reconstructed_img,cv2.COLOR_YUV2BGR)

#cv2.imshow("Reconstructed Image",reconstructed_img)

cv2.waitKey() #Displays images until user hits the 'q' key
cv2.destroyAllWindows() #Makes sure everything is closed



# ~~~~~~~~~~ References ~~~~~~~~~~
'''
https://www.glynholton.com/solutions/exercise-solution-2-17/
https://pythonnumericalmethods.berkeley.edu/notebooks/chapter17.03-Cubic-Spline-Interpolation.html
https://timodenk.com/blog/cubic-spline-interpolation/
'''