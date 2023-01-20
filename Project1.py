import cv2
import numpy as np

img = cv2.imread("sample_image.png") #Read in image
img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV) #Originally BGR, convert to YUV

#Returns the length of each dimension (X pixels, Y pixels, channels)
x_pixels, y_pixels, channels = img.shape

#Grabs each channel from the full colour image as a separate array
luma_img = img[:,:,0] #luma channel (Y), this is the brightness
blue_img = img[:,:,1] #blue channel (U), blue projection
red_img = img[:,:,2] #red channel (V), red projection

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Display orginal image and channels~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cv2.imshow('Full Colour Image',cv2.cvtColor(img,cv2.COLOR_YUV2BGR)) #Display full colour image (needs a conversion)

#Show all three channels separated
separated_channels = np.concatenate((luma_img,blue_img,red_img),axis=1)
cv2.imshow('Y-channel, U-channel, V-channel',separated_channels)


cv2.waitKey() #Displays images until user hits the 'q' key
cv2.destroyAllWindows() #Makes sure everything is closed

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
