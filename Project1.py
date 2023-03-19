"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 4TN4 - Project 1 by Brandon Noble --- Feb. 2023 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is a simple project that takes an input image, compresses it, and then reconstructs the image.
My process for doing so is as follows:
    - Separate the BGR image into YUV colourspace
    - Apply a Gaussian blur (sigma = 0.5) to the Y channel
    - Down sample each channel (Y by 2, U and V by 4) by averaging surrounding pixels
    - Upsample the image with a cubic spline method
    - Apply a Laplacian sharpening filter to the Y channel to enhance edges
    - Convert from YUV to RGB

Using some sample images, the PSNR is ~26.357 on average, accomplishing 87.5% compression.
"""

import cv2
import numpy as np
from time import perf_counter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Function Definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Clip values between 0 and 255
def clip(value):
    return 0 if value < 0 else (255 if value > 255 else value)

#BGR to YUV (coefficients from https://web.archive.org/web/20180423091842/http://www.equasys.de/colorconversion.html)
def BGR_YUV(img):
    x_pixels,y_pixels = img.shape[:2]
    YUV = np.zeros((x_pixels,y_pixels,3),dtype=np.uint8)

    for x in range(x_pixels):
        for y in range(y_pixels):
            #Calculate Y value
            Y = int(0.257*img[x,y,2] + 0.504*img[x,y,1] + 0.098*img[x,y,0] + 16) #Offset added to prevent negative numbers
            YUV[x,y,0] = Y

            #Calculate U value
            U = int(-0.148*img[x,y,2] - 0.291*img[x,y,1] + 0.439*img[x,y,0] + 128) #Offset added to prevent negative numbers
            YUV[x,y,1] = U

            #Calculate Y value
            V = int(0.439*img[x,y,2] - 0.368*img[x,y,1] - 0.071*img[x,y,0] + 128) #Offset added to prevent negative numbers
            YUV[x,y,2] = V
    
    return YUV

#YUV to BGR
def YUV_BGR(img):
    x_pixels,y_pixels = img.shape[:2]
    BGR = np.zeros((x_pixels,y_pixels,3),dtype=np.uint8)

    #Convert the values
    for x in range(x_pixels):
        for y in range(y_pixels):
            #Calculate B value
            B = int(1.164*(img[x,y,0]-16) + 2.017*(img[x,y,1]-128))
            BGR[x,y,0] = clip(B)

            #Calculate G value
            G = int(1.164*(img[x,y,0]-16) - 0.392*(img[x,y,1]-128) - 0.813*(img[x,y,2]-128))
            BGR[x,y,1] = clip(G)

            #Calculate R value
            R = int(1.164*(img[x,y,0]-16) + 1.596*(img[x,y,2]-128))
            BGR[x,y,2] = clip(R)

    return BGR

#Convolution without using premade functions (for 3x3 kernels)
def Convolute_Channel(img_channel,kernel):
    x_pixels,y_pixels = img_channel.shape
    convoluted_channel = img_channel #Save results in a new matrix

    for x in range(1,x_pixels-2):
        for y in range(1,y_pixels-2):
            result = 0.0
            for i in range(0,3):
                for j in range(0,3):
                    result += kernel[i,j] * float(img_channel[x+i-1,y+j-1])
            convoluted_channel[x,y] = clip(result) #Make sure values stay within 0-255

    return convoluted_channel

def Convolute_Full_Image(img,kernel):
    
    #Separate into 3 channels first
    img_channel_0 = img[:,:,0]
    img_channel_1 = img[:,:,1]
    img_channel_2 = img[:,:,2]

    #Convolute filter with each channel before combining
    img_channel_0 = Convolute_Channel(img_channel_0,kernel)
    img_channel_1 = Convolute_Channel(img_channel_1,kernel)
    img_channel_2 = Convolute_Channel(img_channel_2,kernel)

    #Combine the channels
    x_pixels,y_pixels = img_channel_0.shape
    convoluted_image = np.zeros((x_pixels,y_pixels,3),dtype=np.uint8)
    convoluted_image[:,:,0] = img_channel_0
    convoluted_image[:,:,1] = img_channel_1
    convoluted_image[:,:,2] = img_channel_2

    return convoluted_image

#Downsample images
def Down_Sample(img,factor):
    x_pixels,y_pixels = img.shape

    #Check if the image dimensions are odd numbers
    odd_x = True if x_pixels%2 else False
    odd_y = True if y_pixels%2 else False

    #Resize the image if odd
    if odd_x or odd_y:
        new_x = x_pixels+1 if odd_x else x_pixels #If odd X, add 1 to make it even
        new_y = y_pixels+1 if odd_y else y_pixels #If odd Y, add 1 to make it even
        #Resize the image
        img_resize = np.zeros((new_x,new_y),dtype=np.uint8)
        img_resize[:x_pixels-1,:y_pixels-1] = img[:x_pixels-1,:y_pixels-1] #Copy over the original
        
        #Duplicate pixels for the extra row and/or column
        if odd_x:
            img_resize[new_x-1:,:y_pixels-1] = img[x_pixels-1:,:y_pixels-1]
        if odd_y:
            img_resize[:x_pixels-1,new_y-1:] = img[:x_pixels-1,y_pixels-1:]

        #Overwrite input with the resized image
        img = img_resize


    img_comp = np.zeros((x_pixels//factor,y_pixels//factor),dtype=np.uint8) #Creates empty array half the size of the original image

    #Average the surrounding pixels before sampling
    for x in range(0,x_pixels-(factor-1),factor):
        for y in range(0,y_pixels-(factor-1),factor):
            img_comp[x//factor,y//factor] = np.mean(img[x:x+factor,y:y+factor])

    return img_comp

#Cubic function
def Find_Value(abcd,x):
    return clip(abcd[0] * x**3 + abcd[1] * x**2 + abcd[2] * x + abcd[3]) #Prevents lower than 0 and higher than 255

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
def Reconstruct(img_comp,x_pixels,y_pixels,spline_matrix):
    x_pixels_comp, y_pixels_comp = img_comp.shape #size of the compressed Y channel
    
    #If image is odd
    x_pixels = x_pixels+1 if x_pixels%2 else x_pixels
    y_pixels = y_pixels+1 if y_pixels%2 else y_pixels

    comp_factor = x_pixels // x_pixels_comp #factor by which the image was compressed

    img_up = np.zeros((x_pixels+3*comp_factor,y_pixels+3*comp_factor),dtype=np.uint8) #Empty array for the reconstruction

    #Loop through and add all the sampled pixels from the compressed image
    for x in range(0,x_pixels_comp):
        for y in range(0,y_pixels_comp):
            img_up[x*comp_factor,y*comp_factor] = img_comp[x,y]

    #Handle the edges by duplicating last row/column
    img_up[x_pixels-comp_factor:,:] = img_up[x_pixels-4*comp_factor:x_pixels,:]
    img_up[:,y_pixels-comp_factor:] = img_up[:,y_pixels-4*comp_factor:y_pixels]

    #Perform a spline calculation across each row and column, averaging the centre pixels
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Read in and setup images ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

start_time = perf_counter() #Used to find runtime of the program

orig_img = cv2.imread("valley.png") #Read in image into a numpy array
orig_img_yuv = BGR_YUV(orig_img) #Originally BGR, convert to YUV

#Returns the length of each dimension (X pixels, Y pixels, channels)
x_pixels, y_pixels, channels = orig_img_yuv.shape

#Grabs each channel from the full colour image as a separate array
luma_img = orig_img_yuv[:,:,0] #luma channel (Y), this is the brightness
blue_img = orig_img_yuv[:,:,1] #blue channel (U), blue projection
red_img = orig_img_yuv[:,:,2] #red channel (V), red projection

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Display orginal image and channels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cv2.imshow('Full Colour Image',orig_img) #Display full colour image

#Show and save all three channels separated
# separated_channels = np.concatenate((luma_img,blue_img,red_img),axis=1)
# cv2.imshow('Y-channel, U-channel, V-channel',separated_channels)
cv2.imwrite("y_channel_orig.png",luma_img)
cv2.imwrite("u_channel_orig.png",blue_img)
cv2.imwrite("v_channel_orig.png",red_img)

original_size = x_pixels * y_pixels * channels * 8
print("Original image size: " + str(original_size) + " bits")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compress YUV channels  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Apply Gaussian kernel before sampling the Y channel (sigma = 0.5)
gaussian_kernel = np.array([
    [0.0114,0.0842,0.0114],
    [0.0842,0.6194,0.0842],
    [0.0114,0.0842,0.0114]
])

luma_img_blur = Convolute_Channel(luma_img,gaussian_kernel)

#Down sample all the channels
luma_img_comp = Down_Sample(luma_img_blur,2) #compress Y by 2
blue_img_comp = Down_Sample(blue_img,4) #compress U by 4
red_img_comp = Down_Sample(red_img,4) #compress V by 4

#Save down sampled channels
cv2.imwrite("y_channel_comp.png",luma_img_comp)
cv2.imwrite("u_channel_comp.png",blue_img_comp)
cv2.imwrite("v_channel_comp.png",red_img_comp)

#Calculate how much the image was compressed
compressed_size = luma_img_comp.shape[0]*luma_img_comp.shape[1]* 8 + blue_img_comp.shape[0]*blue_img_comp.shape[1]* 8 + red_img_comp.shape[0]*red_img_comp.shape[1]* 8
print("Compressed image size: " + str(compressed_size) + " bits")

print("\tImage compression: " + str(100*(original_size-compressed_size)/original_size) + "%")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Upsample the channels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
spline_matrix = Make_Spline(4) #precalculate for the spline method, will be using 4 pixels x 4 pixels

luma_img_up = Reconstruct(luma_img_comp,x_pixels,y_pixels,spline_matrix) #Y channel
blue_img_up = Reconstruct(blue_img_comp,x_pixels,y_pixels,spline_matrix) #U channel
red_img_up = Reconstruct(red_img_comp,x_pixels,y_pixels,spline_matrix) #V channel

#Save upsampled images
cv2.imwrite("y_channel_up.png",luma_img_up)
cv2.imwrite("u_channel_up.png",blue_img_up)
cv2.imwrite("v_channel_up.png",red_img_up)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reconstruct the full image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Sharpen the image slightly using a Laplacian sharpening filter on the Y channel
#Weights were determined through trial and error
laplacian_kernel = np.array([
    [-0.125,-0.25,-0.125],
    [-0.25,2.5,-0.25],
    [-0.125,-0.25,-0.125],
])

luma_img_up = Convolute_Channel(luma_img_up,laplacian_kernel)

#Create the matrix for the final image
reconstructed_img = np.zeros((x_pixels,y_pixels,3),dtype=np.uint8)

#Add all three channels to one image
reconstructed_img[:,:,0] = luma_img_up[:x_pixels,:y_pixels] #Crop the image to the original size
reconstructed_img[:,:,1] = blue_img_up[:x_pixels,:y_pixels]
reconstructed_img[:,:,2] = red_img_up[:x_pixels,:y_pixels]

reconstructed_img = YUV_BGR(reconstructed_img) #convert back to BGR from YUV

#Display and save image
cv2.imshow("Reconstructed Image",reconstructed_img)
cv2.imwrite("upsampled_image.png",reconstructed_img)

#Calculate PSNR for the luma channel
psnr_value = Get_PSNR(luma_img,luma_img_up[:x_pixels,:y_pixels])
print("PSNR: " + str(psnr_value))

#Calcualtes runtime of the code
runtime_duration = perf_counter()-start_time
print("\nRuntime: " + str(runtime_duration))

#Record the results in a TXT file
with open('results.txt','w') as f:
    f.write("Original image size " + str(original_size) + " bits")
    f.write("\nCompressed image size: " + str(compressed_size) + " bits")
    f.write("\nImage compressed by " + str(100*(original_size-compressed_size)/original_size) + "%")
    f.write("\nPSNR: " + str(psnr_value))
    f.write("\n\nRuntime: " + str(runtime_duration))

cv2.waitKey() #Displays images until user hits the 'q' key
cv2.destroyAllWindows() #Makes sure everything is closed
