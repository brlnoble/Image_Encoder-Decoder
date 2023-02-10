import cv2
import numpy as np
from time import perf_counter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Function Definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Cubic function
def Find_Value(abcd,x):
    return np.clip(abcd[0] * x**3 + abcd[1] * x**2 + abcd[2] * x + abcd[3], 0, 255) #Prevents lower than 0 and higher than 255

#Makes spline coefficient matrix for a given size
def Make_Spline(num_points):
    points =  np.arange(num_points)#The number of points to be included in the spline

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

    #Lastly, invert the matrix so we can use it to solve later on

    return np.linalg.inv(matrix)

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
    return np.dot(spline_mat,z)

#Reconstruct an image using cubic spline
def Reconstruct(img_comp,x_pixels,y_pixels):
    x_pixels_comp, y_pixels_comp = img_comp.shape #size of the compressed Y channel

    comp_factor = x_pixels // x_pixels_comp #factor by which the image was compressed

    img_up = np.zeros((x_pixels+3*comp_factor,y_pixels+3*comp_factor),dtype=np.uint8) #Empty array for the reconstruction

    #Loop through and add all the sampled pixels from the compressed image
    for x in range(0,x_pixels_comp):
        for y in range(0,y_pixels_comp):
            img_up[x*comp_factor,y*comp_factor] = img_comp[x,y]

    #Handle the edges by duplicating last row/column
    img_up[x_pixels:,:] = img_up[x_pixels-3*comp_factor:x_pixels,:]
    img_up[:,y_pixels:] = img_up[:,y_pixels-3*comp_factor:y_pixels]

    #Perform a spline calculation across each row and column, averaging the centre pixels
    spline_matrix = Make_Spline(4) #precalculate for the spline method, will be using 4 pixels x 4 pixels

    for x in range(0,x_pixels,4*(comp_factor-1)):
        for y in range(0,y_pixels,4*(comp_factor-1)):

            for i in range(0,4*comp_factor,comp_factor):
                #Find the spline for the columns
                spline = Find_Spline([img_up[x,y+i],img_up[x+comp_factor,y+i],img_up[x+2*comp_factor,y+i],img_up[x+3*comp_factor,y+i],],spline_matrix)
                
                for j in range(1,3*comp_factor,1):
                    #Skip the known values
                    if not j%comp_factor:
                        continue
                    img_up[x+j,y+i] = Find_Value(spline[4*(j//comp_factor):4*(j//comp_factor+1)],j//comp_factor+0.5)

                #For the rows
                spline = Find_Spline([img_up[x+i,y],img_up[x+i,y+comp_factor],img_up[x+i,y+2*comp_factor],img_up[x+i,y+3*comp_factor],],spline_matrix)
                
                for j in range(1,3*comp_factor,1):
                    #Skip the known values
                    if not j%comp_factor:
                        continue
                    img_up[x+i,y+j] = Find_Value(spline[4*(j//comp_factor):4*(j//comp_factor+1)],j//comp_factor+0.5)

            #Average the centre pixels
            for c in range(0,3*comp_factor,comp_factor):
                for d in range(0,3*comp_factor,comp_factor):
                    for a in range(1,comp_factor):
                        for b in range(1,comp_factor):
                            img_up[x+a+c,y+b+d] = np.uint8(0.25*(
                                int(img_up[x+c,y+b+d]) + int(img_up[x+c+comp_factor,y+b+d]) + int(img_up[x+a+c,y+d]) + int(img_up[x+a+c,y+comp_factor+d])
                            ))

    return img_up

#Calculate PSNR of the reconstructed image
def Get_PSNR(orig_img,img_up):
    #First find mean square error
    mse = np.mean((orig_img/255.0-img_up/255.0)**2)
    #Calculate the PSNR
    return 10*np.log10(1.0/ mse)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Function Definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
start_time = perf_counter()

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

cv2.imshow('Full Colour Image',orig_img) #Display full colour image

#Show all three channels separated
# separated_channels = np.concatenate((luma_img,blue_img,red_img),axis=1)
# cv2.imshow('Y-channel, U-channel, V-channel',separated_channels)

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
blue_shrink = 4
blue_img_comp = np.zeros((x_pixels//blue_shrink,y_pixels//blue_shrink),dtype=np.uint8) #Creates empty array half the size of the original image

for x in range(0,x_pixels-1,blue_shrink):
    for y in range(0,y_pixels-1,blue_shrink):
        blue_img_comp[x//blue_shrink,y//blue_shrink] = blue_img[x,y]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compress V channel by factor of X ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
red_shrink = 4
red_img_comp = np.zeros((x_pixels//red_shrink,y_pixels//red_shrink),dtype=np.uint8) #Creates empty array half the size of the original image

for x in range(0,x_pixels-1,red_shrink):
    for y in range(0,y_pixels-1,red_shrink):
        red_img_comp[x//red_shrink,y//red_shrink] = red_img[x,y]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Display the compressed channels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compressed_size = luma_img_comp.shape[0]*luma_img_comp.shape[1]* 8 + blue_img_comp.shape[0]*blue_img_comp.shape[1]* 8 + red_img_comp.shape[0]*red_img_comp.shape[1]* 8
print("Compressed image size: " + str(compressed_size) + " bits")

print("\tImage compression: " + str(100*(original_size-compressed_size)/original_size) + "%")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Upsample the channels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
luma_img_up = Reconstruct(luma_img_comp,x_pixels,y_pixels) #Y channel
blue_img_up = Reconstruct(blue_img_comp,x_pixels,y_pixels) #U channel
red_img_up = Reconstruct(red_img_comp,x_pixels,y_pixels) #V channel

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reconstruct the full image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# cv2.imwrite("red_reconstruct.png",red_img_up)
# cv2.imwrite("luma_reconstruct_new.png",luma_img_up)

#Combine the three channels
reconstructed_img = np.zeros((x_pixels,y_pixels,3),dtype=np.uint8)

reconstructed_img[:,:,0] = luma_img_up[:x_pixels,:y_pixels] #Crop the image to the original size
reconstructed_img[:,:,1] = blue_img_up[:x_pixels,:y_pixels]
reconstructed_img[:,:,2] = red_img_up[:x_pixels,:y_pixels]

reconstructed_img = cv2.cvtColor(reconstructed_img,cv2.COLOR_YUV2BGR) #convert back to BGR from YUV

cv2.imshow("Reconstructed Image",reconstructed_img)
cv2.imwrite("xyz_reconstruct.png",reconstructed_img)

print("PSNR: " + str(Get_PSNR(orig_img,reconstructed_img)))

print("\nRuntime: " + str(perf_counter()-start_time)) #Calcualtes runtime of the code

cv2.waitKey() #Displays images until user hits the 'q' key
cv2.destroyAllWindows() #Makes sure everything is closed



# ~~~~~~~~~~ References ~~~~~~~~~~
'''
https://www.glynholton.com/solutions/exercise-solution-2-17/
https://pythonnumericalmethods.berkeley.edu/notebooks/chapter17.03-Cubic-Spline-Interpolation.html
https://timodenk.com/blog/cubic-spline-interpolation/
'''
