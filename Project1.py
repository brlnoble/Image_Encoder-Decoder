import cv2
import numpy as np

orig_img = cv2.imread("sample_image.png") #Read in image into a numpy array
orig_img_yuv = cv2.cvtColor(orig_img,cv2.COLOR_BGR2YUV) #Originally BGR, convert to YUV

#Returns the length of each dimension (X pixels, Y pixels, channels)
x_pixels, y_pixels, channels = orig_img_yuv.shape

#Grabs each channel from the full colour image as a separate array
luma_img = orig_img_yuv[:,:,0] #luma channel (Y), this is the brightness
blue_img = orig_img_yuv[:,:,1] #blue channel (U), blue projection
red_img = orig_img_yuv[:,:,2] #red channel (V), red projection

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Display orginal image and channels~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cv2.imshow('Full Colour Image',orig_img) #Display full colour image (needs a conversion)

#Show all three channels separated
separated_channels = np.concatenate((luma_img,blue_img,red_img),axis=1)
#cv2.imshow('Y-channel, U-channel, V-channel',separated_channels)

original_size = x_pixels * y_pixels * channels * 8
print("Original image size: " + str(original_size) + " bits")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Compress Y channel by factor of 2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
luma_shrink = 2 #Factor by which to compress the image
luma_img_comp = np.zeros((x_pixels//luma_shrink,y_pixels//luma_shrink),dtype=np.uint8) #Creates empty array half the size of the original image

for x in range(0,x_pixels-1,luma_shrink):
    for y in range(0,y_pixels-1,luma_shrink):
        luma_img_comp[x//luma_shrink,y//luma_shrink] = luma_img[x,y]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Compress U channel by factor of X~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
blue_shrink = 2
blue_img_comp = np.zeros((x_pixels//blue_shrink,y_pixels//blue_shrink),dtype=np.uint8) #Creates empty array half the size of the original image

for x in range(0,x_pixels-1,blue_shrink):
    for y in range(0,y_pixels-1,blue_shrink):
        blue_img_comp[x//blue_shrink,y//blue_shrink] = blue_img[x,y]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Compress V channel by factor of X~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
red_shrink = 2
red_img_comp = np.zeros((x_pixels//red_shrink,y_pixels//red_shrink),dtype=np.uint8) #Creates empty array half the size of the original image

for x in range(0,x_pixels-1,red_shrink):
    for y in range(0,y_pixels-1,red_shrink):
        red_img_comp[x//red_shrink,y//red_shrink] = red_img[x,y]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Display the compressed channels~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#cv2.imshow("Luma (Y) Channel Shrunk by Factor of "+str(luma_shrink),luma_img_comp)
#cv2.imshow("Blue (U) Channel Shrunk by Factor of "+str(blue_shrink),blue_img_comp)
#cv2.imshow("Red (U) Channel Shrunk by Factor of "+str(red_shrink),red_img_comp)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Reconstruct the Y channel~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_pixels_comp, y_pixels_comp = luma_img_comp.shape #size of the compressed Y channel

comp_factor = x_pixels // x_pixels_comp #factor by which the image was compressed

luma_img_up = np.zeros((x_pixels,y_pixels),dtype=np.uint8) #Empty array for the reconstruction

#Loop through and add all the sampled pixels from the compressed image
for x in range(0,x_pixels_comp):
    for y in range(0,y_pixels_comp):
        luma_img_up[x*comp_factor,y*comp_factor] = luma_img_comp[x,y]

#Interpret the centre pixels between each set of 4
for x in range(1,x_pixels-1,2):
    for y in range(1,y_pixels-1,2):
        luma_img_up[x,y] = sum([luma_img_up[x-1,y-1],luma_img_up[x-1,y+1],luma_img_up[x+1,y-1],luma_img_up[x+1,y+1]])//4

#Interpret the next set of pixels
for x in range(2,x_pixels-2,2):
    for y in range(1,y_pixels-2,2):
        luma_img_up[x,y] = sum([luma_img_up[x-1,y],luma_img_up[x,y+1],luma_img_up[x+1,y],luma_img_up[x-1,y]])//4

for x in range(1,x_pixels-2,2):
    for y in range(2,y_pixels-2,2):
        luma_img_up[x,y] = sum([luma_img_up[x-1,y],luma_img_up[x,y+1],luma_img_up[x+1,y],luma_img_up[x-1,y]])//4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Reconstruct the U channel~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_pixels_comp, y_pixels_comp = blue_img_comp.shape #size of the compressed Y channel

comp_factor = x_pixels // x_pixels_comp #factor by which the image was compressed

blue_img_up = np.zeros((x_pixels,y_pixels),dtype=np.uint8) #Empty array for the reconstruction

#Loop through and add all the sampled pixels from the compressed image
for x in range(0,x_pixels_comp):
    for y in range(0,y_pixels_comp):
        blue_img_up[x*comp_factor,y*comp_factor] = blue_img_comp[x,y]

#Interpret the centre pixels between each set of 4
for x in range(1,x_pixels-1,2):
    for y in range(1,y_pixels-1,2):
        blue_img_up[x,y] = sum([blue_img_up[x-1,y-1],blue_img_up[x-1,y+1],blue_img_up[x+1,y-1],blue_img_up[x+1,y+1]])//4

#Interpret the next set of pixels
for x in range(2,x_pixels-2,2):
    for y in range(1,y_pixels-2,2):
        blue_img_up[x,y] = sum([blue_img_up[x-1,y],blue_img_up[x,y+1],blue_img_up[x+1,y],blue_img_up[x-1,y]])//4

for x in range(1,x_pixels-2,2):
    for y in range(2,y_pixels-2,2):
        blue_img_up[x,y] = sum([blue_img_up[x-1,y],blue_img_up[x,y+1],blue_img_up[x+1,y],blue_img_up[x-1,y]])//4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Reconstruct the V channel~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_pixels_comp, y_pixels_comp = red_img_comp.shape #size of the compressed Y channel

comp_factor = x_pixels // x_pixels_comp #factor by which the image was compressed

red_img_up = np.zeros((x_pixels,y_pixels),dtype=np.uint8) #Empty array for the reconstruction

#Loop through and add all the sampled pixels from the compressed image
for x in range(0,x_pixels_comp):
    for y in range(0,y_pixels_comp):
        red_img_up[x*comp_factor,y*comp_factor] = red_img_comp[x,y]

#Interpret the centre pixels between each set of 4
for x in range(1,x_pixels-1,2):
    for y in range(1,y_pixels-1,2):
        red_img_up[x,y] = sum([red_img_up[x-1,y-1],red_img_up[x-1,y+1],red_img_up[x+1,y-1],red_img_up[x+1,y+1]])//4

#Interpret the next set of pixels
for x in range(2,x_pixels-2,2):
    for y in range(1,y_pixels-2,2):
        red_img_up[x,y] = sum([red_img_up[x-1,y],red_img_up[x,y+1],red_img_up[x+1,y],red_img_up[x-1,y]])//4

for x in range(1,x_pixels-2,2):
    for y in range(2,y_pixels-2,2):
        red_img_up[x,y] = sum([red_img_up[x-1,y],red_img_up[x,y+1],red_img_up[x+1,y],red_img_up[x-1,y]])//4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Reconstruct the full image~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#cv2.imshow("Luma Reconstruct",luma_img_up)
#cv2.imshow("Blue Reconstruct",blue_img_up)
#cv2.imshow("Red Reconstruct",red_img_up)

reconstructed_img = np.zeros((x_pixels,y_pixels,3),dtype=np.uint8)

reconstructed_img[:,:,0] = luma_img_up
reconstructed_img[:,:,1] = blue_img_up
reconstructed_img[:,:,2] = red_img_up

reconstructed_img = cv2.cvtColor(reconstructed_img,cv2.COLOR_YUV2BGR)

cv2.imshow("Reconstructed Image",reconstructed_img)

cv2.waitKey() #Displays images until user hits the 'q' key
cv2.destroyAllWindows() #Makes sure everything is closed
