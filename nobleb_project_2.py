import numpy as np #for numpy arrays
import cv2 #for image read/write
from time import perf_counter #for finding runtime
from skimage.metrics import structural_similarity as SSIM #for SSIM calculation

from tensorflow.keras.models import * #load CNN model and perform predictions

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Function Definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Clip values between 0 and 255
def clip(value):
    return 0 if value < 0 else (255 if value > 255 else value)

def Clip_Image(img):
    x_pixels,y_pixels = img.shape
    result = np.zeros((x_pixels,y_pixels),dtype=np.uint8)

    for x in range(x_pixels-1):
        for y in range(y_pixels-1):
            result[x,y] = clip(img[x,y])

    return result

#Pad image to be the right size
def Pad_Image(size,img):
    x_pixels,y_pixels = img.shape

    #Return current image if already the right size
    if x_pixels == size[0] and y_pixels == size[1]:
        return img

    #Check if the image dimensions are odd numbers
    odd_x = True if x_pixels < size[0] else False
    odd_y = True if y_pixels < size[1] else False

    #Resize the image if odd
    if odd_x or odd_y:
        new_x = x_pixels+1 if odd_x else x_pixels #If odd X, add 1 to make it even
        new_y = y_pixels+1 if odd_y else y_pixels #If odd Y, add 1 to make it even
        #Resize the image
        img_resize = np.zeros((new_x,new_y),dtype=np.uint8)
        img_resize[:x_pixels,:y_pixels] = img[:x_pixels,:y_pixels] #Copy over the original
        
        #Duplicate pixels for the extra row and/or column
        if odd_x:
            img_resize[-1,:] = img_resize[-2,:]
        if odd_y:
            img_resize[:,-1] = img_resize[:,-2]

        #Overwrite input with the resized image
        img = img_resize

        #See if the result should be redone
        x_pixels,y_pixels = img.shape
        if x_pixels < size[0] or y_pixels < size[1]:
            img = Pad_Image(size,img)

    return img

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

#Calculate PSNR of the reconstructed image
def Get_PSNR(orig_img,img_up):
    #First find mean square error
    mse = np.mean((orig_img/255.0-img_up/255.0)**2)
    #Calculate the PSNR
    return 10*np.log10(1.0/ mse)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Read in and setup images ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

start_timer = perf_counter()

img = cv2.imread("valley.png")
x_pixels,y_pixels,channels = img.shape #size of the original image

original_size = x_pixels * y_pixels * channels * 8
print("Original image size: " + str(original_size) + " bits")

#Separate into the YUV channels
img_yuv = BGR_YUV(img) #Convert to YUV colourspace
luma_img = img_yuv[:,:,0]
blue_img = img_yuv[:,:,1]
red_img = img_yuv[:,:,2]

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

#Calculate how much the image was compressed
compressed_size = luma_img_comp.shape[0]*luma_img_comp.shape[1]* 8 + blue_img_comp.shape[0]*blue_img_comp.shape[1]* 8 + red_img_comp.shape[0]*red_img_comp.shape[1]* 8
print("Compressed image size: " + str(compressed_size) + " bits")

print("\tImage compression: " + str(100*(original_size-compressed_size)/original_size) + "%")

#Save the steps
cv2.imwrite("y_channel_orig.png",luma_img)
cv2.imwrite("u_channel_orig.png",blue_img)
cv2.imwrite("v_channel_orig.png",red_img)

cv2.imwrite("y_channel_comp.png",luma_img_comp)
cv2.imwrite("u_channel_comp.png",blue_img_comp)
cv2.imwrite("v_channel_comp.png",red_img_comp)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Upsample the channels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Load in the model
upsample_model = load_model("nobleb_cnn.h5")

#Redimension the arrays for use with the model
luma_img_comp = np.expand_dims(luma_img_comp,axis=0)
blue_img_comp = np.expand_dims(blue_img_comp,axis=0)
red_img_comp = np.expand_dims(red_img_comp,axis=0)

#Use the model to predict the output
luma_img_up = np.array(upsample_model.predict(luma_img_comp))
luma_img_up = luma_img_up[0][:,:,0] #Remove extra dimensions from the upsampled result

blue_img_up = np.array(upsample_model.predict(blue_img_comp))
blue_img_up = np.array(upsample_model.predict(blue_img_up)) #Perform twice to upscale by a factor of 4
blue_img_up = blue_img_up[0][:,:,0] #Remove extra dimensions from the upsampled result

red_img_up = np.array(upsample_model.predict(red_img_comp))
red_img_up = np.array(upsample_model.predict(red_img_up)) #Perform twice to upscale by a factor of 4
red_img_up = red_img_up[0][:,:,0] #Remove extra dimensions from the upsampled result

#Clip values to be integers from 0 - 255
luma_img_up = Clip_Image(luma_img_up)
blue_img_up = Clip_Image(blue_img_up)
red_img_up = Clip_Image(red_img_up)

luma_img_up = luma_img_up[:x_pixels-1,:y_pixels-1]
blue_img_up = blue_img_up[:x_pixels-1,:y_pixels-1]
red_img_up = red_img_up[:x_pixels-1,:y_pixels-1]

# #Pad the images if they are not the correct size
luma_img_up = Pad_Image([x_pixels,y_pixels],luma_img_up)
blue_img_up = Pad_Image([x_pixels,y_pixels],blue_img_up)
red_img_up = Pad_Image([x_pixels,y_pixels],red_img_up)

#Combine the images and convert to BGR
up_img = np.zeros((x_pixels,y_pixels,3),dtype=np.uint8)
up_img[:,:,0] = luma_img_up[:x_pixels,:y_pixels]
up_img[:,:,1] = blue_img_up[:x_pixels,:y_pixels]
up_img[:,:,2] = red_img_up[:x_pixels,:y_pixels]

#Sharpen the luma channel slightly to reduce blur
laplacian_kernel = np.array([
    [-0.125,-0.25,-0.125],
    [-0.25,2.5,-0.25],
    [-0.125,-0.25,-0.125],
])

up_img[:,:,0] = Convolute_Channel(up_img[:,:,0],laplacian_kernel)

#Save the resultant images
cv2.imwrite("y_channel_up.png",up_img[:,:,0])
cv2.imwrite("u_channel_up.png",up_img[:,:,1])
cv2.imwrite("v_channel_up.png",up_img[:,:,2])

up_img = YUV_BGR(up_img)

cv2.imshow("Original",img)
cv2.imshow("Upsampled",up_img)

#cv2.imshow("YUV",np.concatenate((luma_img_up,blue_img_up,red_img_up),axis=0))

#Calculate the PSNR and SSIM
psnr_value = Get_PSNR(luma_img,luma_img_up[:x_pixels,:y_pixels])
ssim_value = SSIM(img,up_img,channel_axis=2)
print("PSNR: " + str(psnr_value))
print("SSIM: " + str(ssim_value))

runtime_duration = perf_counter() - start_timer
print("\nRuntime: " + str(runtime_duration))

#Record the results in a TXT file
with open('results.txt','w') as f:
    f.write("Original image size " + str(original_size) + " bits")
    f.write("\nCompressed image size: " + str(compressed_size) + " bits")
    f.write("\nImage compressed by " + str(100*(original_size-compressed_size)/original_size) + "%")
    f.write("\nPSNR: " + str(psnr_value))
    f.write("\nSSIM: " + str(ssim_value))
    f.write("\n\nRuntime: " + str(runtime_duration))

cv2.imwrite("upsampled.png",up_img)

cv2.waitKey(0)
cv2.destroyAllWindows()