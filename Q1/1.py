import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import math
import scipy

def imshow(image):
    if len(image.shape) == 3:
      # Height, width, channels
      pass
    else:
      # Height, width - must be grayscale
      # convert to RGB, since matplotlib will plot in a weird colormap (instead of black = 0, white = 1)
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Draw the image
    plt.imshow(image)
    # Disable drawing the axes and tick marks in the plot, since it's actually an image
    plt.axis('off')
    plt.show()

n = int(input('Enter n:'))
m = int(input('Enter m:'))    
select_img = int(input('Enter 1 for Barbara, 2 for Cat:'))
if select_img == 1:
    input_img = cv2.imread('1__M_Right_little_finger.jpg')
    img_name = 'fingerprint'
elif select_img == 2:
    input_img = cv2.imread('./cat.jpg')
    img_name = 'cat'
input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
# imshow(input_img_rgb)

def keep_n_per_block(img, n):
    x_lim = img.shape[0]
    y_lim = img.shape[1]
    result_img = np.zeros((x_lim, y_lim))

    for j in range(0, y_lim, 8):
        for i in range(0, x_lim, 8):
            result_img[i:(i+n), j:(j+n)] = img[i:(i+n), j:(j+n)]
    return result_img


#(a)

#COMPRESS
rgb_quantization_table = np.array([[ 8, 6, 6, 7, 6, 5, 8, 7],
                                   [ 7, 7, 9, 9, 8,10,12,20],
                                   [13,12,11,11,12,25,18,19],
                                   [15,20,29,26,31,30,29,26],
                                   [28,28,32,36,46,39,32,34],
                                   [44,35,28,28,40,55,41,44],
                                   [48,49,52,52,52,31,39,57],
                                   [61,56,50,60,46,51,52,50]])
    

input_img_rgb_float = input_img_rgb.astype(np.float32)
x_lim = input_img_rgb.shape[0]
y_lim = input_img_rgb.shape[1]

#divide the img to 3 channels
input_img_r_float = np.zeros((x_lim, y_lim))
input_img_g_float = np.zeros((x_lim, y_lim))
input_img_b_float = np.zeros((x_lim, y_lim))
for j in range(0, y_lim):
    for i in range(0, x_lim):
        input_img_r_float[i, j] = input_img_rgb_float[i, j][0]
        input_img_g_float[i, j] = input_img_rgb_float[i, j][1]
        input_img_b_float[i, j] = input_img_rgb_float[i, j][2]

#DCT for each channel
def dct2d(img):
    # Get image dimensions
    height, width = img.shape
    # Initialize output array
    dct = np.zeros((height, width))
    # Compute coefficients
    for i in range(height):
        for j in range(width):
            sum = 0
            for m in range(height):
                for n in range(width):
                    sum += img[m, n] * math.cos(math.pi * (2 * m + 1) * i / (2 * height)) * math.cos(math.pi * (2 * n + 1) * j / (2 * width))
            sum *= math.sqrt(2 / height) * math.sqrt(2 / width)
            if i == 0:
                sum *= 1 / math.sqrt(2)
            if j == 0:
                sum *= 1 / math.sqrt(2)
            dct[i, j] = sum
    return dct

def idct2d(dct):
    # Get DCT dimensions
    height, width = dct.shape
    # Initialize output array
    img = np.zeros((height, width))
    # Compute pixel values
    for m in range(height):
        for n in range(width):
            sum = 0
            for i in range(height):
                for j in range(width):
                    ci = 1 / math.sqrt(2) if i == 0 else 1
                    cj = 1 / math.sqrt(2) if j == 0 else 1
                    sum += ci * cj / 4 * dct[i, j] * math.cos(math.pi * (2 * m + 1) * i / (2 * height)) * math.cos(math.pi * (2 * n + 1) * j / (2 * width))
            img[m, n] = sum
    return img

dct_img_r = np.zeros((x_lim, y_lim))
dct_img_g = np.zeros((x_lim, y_lim))
dct_img_b = np.zeros((x_lim, y_lim))

for j in range(0, y_lim, 8):
    for i in range(0, x_lim, 8):
        dct_img_r[i:(i+8), j:(j+8)] = dct2d(input_img_r_float[i:(i+8), j:(j+8)])
        dct_img_g[i:(i+8), j:(j+8)] = dct2d(input_img_g_float[i:(i+8), j:(j+8)])
        dct_img_b[i:(i+8), j:(j+8)] = dct2d(input_img_b_float[i:(i+8), j:(j+8)])

dct_img_r = keep_n_per_block(dct_img_r, n)
dct_img_g = keep_n_per_block(dct_img_g, n)
dct_img_b = keep_n_per_block(dct_img_b, n)

#Quantize using quantization table
for j in range(0, y_lim, 8):
    for i in range(0, x_lim, 8):
        for y in range(0, 8):
            for x in range(0, 8):
                dct_img_r[(i+x), (j+y)] = round(dct_img_r[(i+x), (j+y)]/rgb_quantization_table[x, y])
                dct_img_g[(i+x), (j+y)] = round(dct_img_g[(i+x), (j+y)]/rgb_quantization_table[x, y])
                dct_img_b[(i+x), (j+y)] = round(dct_img_b[(i+x), (j+y)]/rgb_quantization_table[x, y])

# plt.imshow(dct_img_r)
# plt.show()
# plt.imshow(dct_img_g)
# plt.show()
# plt.imshow(dct_img_b)
# plt.show()

#uniform quantization in m bits
max_value = max(np.max(dct_img_r), np.max(dct_img_g), np.max(dct_img_b))
min_value = min(np.min(dct_img_r), np.min(dct_img_g), np.min(dct_img_b))
step_size = (max_value - min_value) / 2**m
offset = round(min_value / step_size)
# print(max_value, min_value, step_size)

for j in range(0, y_lim):
    for i in range(0, x_lim):
        dct_img_r[i, j] = round(dct_img_r[i, j]/ step_size) - offset
        dct_img_g[i, j] = round(dct_img_g[i, j]/ step_size) - offset
        dct_img_b[i, j] = round(dct_img_b[i, j]/ step_size) - offset

# plt.imshow(dct_img_r)
# plt.show()

#combine 3 channels to 1 img
dct_img_rgb = np.zeros((x_lim, y_lim, 3))
# dct_img_rgb = input_img_rgb.copy() #just to get array with same size
for j in range(0, y_lim):
    for i in range(0, x_lim):
        dct_img_rgb[i, j][0] = dct_img_r[i, j]
        dct_img_rgb[i, j][1] = dct_img_g[i, j]
        dct_img_rgb[i, j][2] = dct_img_b[i, j]


#DECOMPRESS

#divide the img to 3 channels
idct_img_r = np.zeros((x_lim, y_lim))
idct_img_g = np.zeros((x_lim, y_lim))
idct_img_b = np.zeros((x_lim, y_lim))
for j in range(0, y_lim):
    for i in range(0, x_lim):
        idct_img_r[i, j] = dct_img_rgb[i, j][0]
        idct_img_g[i, j] = dct_img_rgb[i, j][1]
        idct_img_b[i, j] = dct_img_rgb[i, j][2]

#Uniform Unquantization
for j in range(0, y_lim):
    for i in range(0, x_lim):
        idct_img_r[i, j] = (idct_img_r[i, j] + offset)*step_size
        idct_img_g[i, j] = (idct_img_g[i, j] + offset)*step_size
        idct_img_b[i, j] = (idct_img_b[i, j] + offset)*step_size

#Unquantize using quantization table
for j in range(0, y_lim, 8):
    for i in range(0, x_lim, 8):
        for y in range(0, 8):
            for x in range(0, 8):
                idct_img_r[(i+x), (j+y)] = idct_img_r[(i+x), (j+y)]*rgb_quantization_table[x, y]
                idct_img_g[(i+x), (j+y)] = idct_img_g[(i+x), (j+y)]*rgb_quantization_table[x, y]
                idct_img_b[(i+x), (j+y)] = idct_img_b[(i+x), (j+y)]*rgb_quantization_table[x, y]

#IDCT for each channel
result_img_r = np.zeros((x_lim, y_lim))
result_img_g = np.zeros((x_lim, y_lim))
result_img_b = np.zeros((x_lim, y_lim))

for j in range(0, y_lim, 8):
    for i in range(0, x_lim, 8):
        result_img_r[i:(i+8), j:(j+8)] = idct2d(idct_img_r[i:(i+8), j:(j+8)])
        result_img_g[i:(i+8), j:(j+8)] = idct2d(idct_img_g[i:(i+8), j:(j+8)])
        result_img_b[i:(i+8), j:(j+8)] = idct2d(idct_img_b[i:(i+8), j:(j+8)])

#combine 3 channels to 1 img
result_img_rgb = input_img_rgb.copy() #just to make array with same size
for j in range(0, y_lim):
    for i in range(0, x_lim):
        R = result_img_r[i, j]
        G = result_img_g[i, j]
        B = result_img_b[i, j]
        #prevent overflow or underflow
        result_img_rgb[i, j][0] = R if R <= 255 and R >= 0 else 255 if R > 255 else 0 #R part
        result_img_rgb[i, j][1] = G if G <= 255 and G >= 0 else 255 if G > 255 else 0 #G part
        result_img_rgb[i, j][2] = B if B <= 255 and B >= 0 else 255 if B > 255 else 0 #B part

imshow(result_img_rgb)
cv2.imwrite('./output/'+img_name+'_n'+str(n)+'m'+str(m)+'_a.jpg', cv2.cvtColor(result_img_rgb, cv2.COLOR_RGB2BGR))

#compute compression ratios
compress_ratio = (x_lim*y_lim*8*3)/(x_lim*y_lim*m)
print('Compression ratio:', compress_ratio)

#compute PSNR values
mse = np.mean((input_img_rgb.astype(np.float32)-result_img_rgb.astype(np.float32))**2)
max_pixel = 255.0
psnr = 20*math.log10(max_pixel / math.sqrt(mse))
print('PSNR value:', psnr)




#(b)

input_img_ycc = input_img_rgb.copy()
x_lim = input_img_ycc.shape[0]
y_lim = input_img_ycc.shape[1]

#convert input img from RGB color space to YCbCr color space
for j in range(0, y_lim):
   for i in range(0, x_lim):
        input_img_ycc[i, j][0] = 16+ 0.257*input_img_rgb[i, j][0] + 0.564*input_img_rgb[i, j][1] + 0.098*input_img_rgb[i, j][2] #Y part
        if j%2 == 0 and i%2 == 0: #sample chrominance
            Cb = 128 - 0.148*input_img_rgb[i, j][0] - 0.291*input_img_rgb[i, j][1] + 0.439*input_img_rgb[i, j][2]
            Cr = 128 + 0.439*input_img_rgb[i, j][0] - 0.368*input_img_rgb[i, j][1] - 0.071*input_img_rgb[i, j][2]
            #4:2:0 chrominance subsampling
            for y in range(j, j+2):
               for x in range(i, i+2):
                    if y < y_lim and x < x_lim:
                        input_img_ycc[x, y][1] = Cb
                        input_img_ycc[x, y][2] = Cr


#COMPRESS
lumin_quantization_table = np.array([[16,11,10,16,24,40,51,61],
                                     [12,12,14,19,26,58,60,55],
                                     [14,13,16,24,40,57,69,56],
                                     [14,17,22,29,51,87,80,62],
                                     [18,22,37,56,68,109,103,77],
                                     [24,36,55,64,81,104,113,92],
                                     [49,64,78,87,103,121,120,101],
                                     [72,92,95,98,112,100,103,99]])

chrom_quantization_table = np.array([[17,18,24,47,99,99,99,99],
                                     [18,21,26,66,99,99,99,99],
                                     [24,26,56,99,99,99,99,99],
                                     [47,66,99,99,99,99,99,99],
                                     [99,99,99,99,99,99,99,99],
                                     [99,99,99,99,99,99,99,99],
                                     [99,99,99,99,99,99,99,99],
                                     [99,99,99,99,99,99,99,99]])

#compress input_img_ycc
input_img_ycc_float = input_img_ycc.astype(np.float32)

#divide the img to 3 channels
input_img_y_float = np.zeros((x_lim, y_lim))
input_img_cb_float = np.zeros((x_lim, y_lim))
input_img_cr_float = np.zeros((x_lim, y_lim))
for j in range(0, y_lim):
    for i in range(0, x_lim):
        input_img_y_float[i, j] = input_img_ycc_float[i, j][0]
        input_img_cb_float[i, j] = input_img_ycc_float[i, j][1]
        input_img_cr_float[i, j] = input_img_ycc_float[i, j][2]

#DCT for each channel
dct_img_y = np.zeros((x_lim, y_lim))
dct_img_cb = np.zeros((x_lim, y_lim))
dct_img_cr = np.zeros((x_lim, y_lim))

for j in range(0, y_lim, 8):
    for i in range(0, x_lim, 8):
        dct_img_y[i:(i+8), j:(j+8)] = dct2d(input_img_y_float[i:(i+8), j:(j+8)])
        dct_img_cb[i:(i+8), j:(j+8)] = dct2d(input_img_cb_float[i:(i+8), j:(j+8)])
        dct_img_cr[i:(i+8), j:(j+8)] = dct2d(input_img_cr_float[i:(i+8), j:(j+8)])

dct_img_y = keep_n_per_block(dct_img_y, n)
dct_img_cb = keep_n_per_block(dct_img_cb, n)
dct_img_cr = keep_n_per_block(dct_img_cr, n)

#Quantize using quantization table
for j in range(0, y_lim, 8):
    for i in range(0, x_lim, 8):
        for y in range(0, 8):
            for x in range(0, 8):
                dct_img_y[(i+x), (j+y)] = round(dct_img_y[(i+x), (j+y)]/lumin_quantization_table[x, y])
                dct_img_cb[(i+x), (j+y)] = round(dct_img_cb[(i+x), (j+y)]/chrom_quantization_table[x, y])
                dct_img_cr[(i+x), (j+y)] = round(dct_img_cr[(i+x), (j+y)]/chrom_quantization_table[x, y])

# plt.imshow(dct_img_y)
# plt.show()

#uniform quantization in m bits
max_value_y = np.max(dct_img_y)
max_value_cc = max(np.max(dct_img_cb), np.max(dct_img_cr))

min_value_y = np.min(dct_img_y)
min_value_cc = min(np.min(dct_img_cb), np.min(dct_img_cr))

step_size_y = (max_value_y - min_value_y) / 2**m
step_size_cc = (max_value_cc - min_value_cc) / 2**m

offset_y = round(min_value_y / step_size_y)
offset_cc = round(min_value_cc / step_size_cc)

# print(max_value_y, min_value_y, step_size_y)
# print(max_value_cb, min_value_cb, step_size_cb)
# print(max_value_cr, min_value_cr, step_size_cr)

for j in range(0, y_lim):
    for i in range(0, x_lim):
        dct_img_y[i, j] = round(dct_img_y[i, j]/ step_size_y) - offset_y
        dct_img_cb[i, j] = round(dct_img_cb[i, j]/ step_size_cc) - offset_cc
        dct_img_cr[i, j] = round(dct_img_cr[i, j]/ step_size_cc) - offset_cc

# plt.imshow(dct_img_r)
# plt.show()

#combine 3 channels to 1 img
dct_img_ycc = np.zeros((x_lim, y_lim, 3))
# dct_img_ycc = input_img_rgb.copy() #just to get array with same size
for j in range(0, y_lim):
    for i in range(0, x_lim):
        dct_img_ycc[i, j][0] = dct_img_y[i, j]
        dct_img_ycc[i, j][1] = dct_img_cb[i, j]
        dct_img_ycc[i, j][2] = dct_img_cr[i, j]

#DECOMPRESS
#divide the img to 3 channels
idct_img_y = np.zeros((x_lim, y_lim))
idct_img_cb = np.zeros((x_lim, y_lim))
idct_img_cr = np.zeros((x_lim, y_lim))
for j in range(0, y_lim):
    for i in range(0, x_lim):
        idct_img_y[i, j] = dct_img_ycc[i, j][0]
        idct_img_cb[i, j] = dct_img_ycc[i, j][1]
        idct_img_cr[i, j] = dct_img_ycc[i, j][2]

#Uniform Unquantization
for j in range(0, y_lim):
    for i in range(0, x_lim):
        idct_img_y[i, j] = (idct_img_y[i, j] + offset_y)*step_size_y
        idct_img_cb[i, j] = (idct_img_cb[i, j] + offset_cc)*step_size_cc
        idct_img_cr[i, j] = (idct_img_cr[i, j] + offset_cc)*step_size_cc


#Unquantize using quantization table
for j in range(0, y_lim, 8):
    for i in range(0, x_lim, 8):
        for y in range(0, 8):
            for x in range(0, 8):
                idct_img_y[(i+x), (j+y)] = idct_img_y[(i+x), (j+y)]*lumin_quantization_table[x, y]
                idct_img_cb[(i+x), (j+y)] = idct_img_cb[(i+x), (j+y)]*chrom_quantization_table[x, y]
                idct_img_cr[(i+x), (j+y)] = idct_img_cr[(i+x), (j+y)]*chrom_quantization_table[x, y]


#IDCT for each channel
result_img_y = np.zeros((x_lim, y_lim))
result_img_cb = np.zeros((x_lim, y_lim))
result_img_cr = np.zeros((x_lim, y_lim))
for j in range(0, y_lim, 8):
    for i in range(0, x_lim, 8):
        result_img_y[i:(i+8), j:(j+8)] = idct2d(idct_img_y[i:(i+8), j:(j+8)])
        result_img_cb[i:(i+8), j:(j+8)] = idct2d(idct_img_cb[i:(i+8), j:(j+8)])
        result_img_cr[i:(i+8), j:(j+8)] = idct2d(idct_img_cr[i:(i+8), j:(j+8)])



#convert img from YCbCr color space to RGB color space
converted_img = input_img_rgb.copy() #just to make array with same shape

for j in range(0, y_lim):
   for i in range(0, x_lim):
        R = 1.164*(result_img_y[i, j]-16) + 0*(result_img_cb[i, j]-128) + 1.596*(result_img_cr[i, j]-128)
        G = 1.164*(result_img_y[i, j]-16) - 0.382*(result_img_cb[i, j]-128) - 0.813*(result_img_cr[i, j]-128)
        B = 1.164*(result_img_y[i, j]-16) + 2.017*(result_img_cb[i, j]-128) + 0*(result_img_cr[i, j]-128)
        #prevent overflow or underflow
        converted_img[i, j][0] = R if R <= 255 and R >= 0 else 255 if R > 255 else 0 #R part
        converted_img[i, j][1] = G if G <= 255 and G >= 0 else 255 if G > 255 else 0 #G part
        converted_img[i, j][2] = B if B <= 255 and B >= 0 else 255 if B > 255 else 0 #B part

imshow(converted_img)
cv2.imwrite('./output/'+img_name+'_n'+str(n)+'m'+str(m)+'_b.jpg', cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR))

#compute compression ratios
compress_ratio = (x_lim*y_lim*8*3)/(x_lim*y_lim*m)
print('Compression ratio:', compress_ratio)

#compute PSNR values
mse = np.mean((input_img_rgb.astype("float")-converted_img.astype("float"))**2)
max_pixel = 255.0
psnr = 20*math.log10(max_pixel / math.sqrt(mse))
print('PSNR value:', psnr)