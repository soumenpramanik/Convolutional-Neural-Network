import numpy as np
import cv2 as cv
def patch_extraction(input_tensor, kh, kw, sh, sw):
    assert(len(input_tensor.shape)==3), "Input should 3d, we got " + str(len(input_tensor.shape))
    h_s = int(kh/2.0)
    w_s = int(kw/2.0)
    h_e = input_tensor.shape[0] - h_s
    w_e = input_tensor.shape[1] - w_s
    shape = (int((h_e-h_s)/sh), int((w_e-w_s)/sw))
    output_patches = np.zeros((int((h_e-h_s)/sh), int((w_e-w_s)/sw), kh, kw, input_tensor.shape[2]))
    for h_c,h in enumerate(range(h_s, h_e, sh)):
        for w_c, w in enumerate(range(w_s, w_e, sw)):
            output_patches[h_c, w_c, :, :,:] = input_tensor[h-h_s:h+h_s+1, w-w_s:w+w_s+1, :]
            #print (output_patches.shape)
    output_patches = output_patches.reshape(int((h_e-h_s)/sh)*int((w_e-w_s)/sw), kh*kw*input_tensor.shape[2])
    #print (output_patches.shape)
    #output_patches = output_patches.reshape(int((h_e-h_s)/sh)*int((w_e-w_s)/sw), kh*kw*input_tensor.shape[2])
    return output_patches, shape
#img = np.random.rand(150,300,3)out_h
#patches,shape = patch_extraction(img, 3,3,1,1)
#print(patches.shape)
#import sklearn.feature_extraction.image as image

def convolution_1kernel(input_tensor, kernel, stride, pad):
    #out_h = (input_tensor.shape[0] + 2*pad[0] - kernel.shape[0])/stride[0] + 1
    #out_w =  (input_tensor.shape[1] + 2*pad[1] - kernel.shape[1])/stride[1] + 1
    padded_input = np.zeros((input_tensor.shape[0] + 2*pad[0], input_tensor.shape[1] + 2*pad[1], input_tensor.shape[2]))
    padded_input[kernel.shape[0]/2:input_tensor.shape[0]+kernel.shape[0]/2, kernel.shape[1]/2:input_tensor.shape[1]+kernel.shape[1]/2, :] = input_tensor
    #print (padded_input.shape)
    patches, shape = patch_extraction(padded_input, kernel.shape[0], kernel.shape[1], stride[0], stride[1])

    kernel = kernel.reshape(kernel.shape[0]*kernel.shape[1]*kernel.shape[2])
    ouput_feature_map = np.zeros((patches.shape[0], 1))
    for i in range(ouput_feature_map.shape[0]):
        ouput_feature_map[i] = np.dot(patches[i], kernel)

    ouput_feature_map = ouput_feature_map.reshape(shape[0], shape[1])

    return ouput_feature_map

#img = np.random.rand(150,300,3)
#kernel = np.random.rand(3,3,3)
#stride =(1,1)
#pad = (1,1)
#output_feature_map = convolution_1kernel(img, kernel, stride, pad)
#print (output_feature_map.shape)
#patches,shape = patch_extraction(img, 3,3,1,1)

def conv_2d_full(input_tensor, kernel, stride, pad, output_channels):
    output = [convolution_1kernel(input_tensor, kernel[i], stride, pad) for i in range(output_channels)]
    print(np.asarray(output).shape)
    return np.asarray(output)

edge = True
generic = False
if edge:
    img = cv.imread('DSC01806.png')
    print(img.shape)
    kernel = np.zeros((3,3,3))
    pad = (1,1)
    stride = (1,1)
    print(kernel.shape)
    k = np.array([[0.0, -1, 0],[-1,4,-1],[0,-1,0]]) # Edge detection
    for i in range(3):
        for j in range(3):
            for kk in range(3):
                kernel[i][j][kk] = k[i][j]
    #kernel.dtype = np.float32
    print(kernel.shape)
    print(k.shape)
    img_out = convolution_1kernel(img, kernel, stride, pad)#, 96)
    print (img_out.shape)
    cv.imwrite('DSC01807.JPG', img_out)
if generic:
    img = np.random.rand(332,332,3)
    kernel = np.zeros((96, 3,3,3))
    pad = (1,1)
    stride = (1,1)
    conv_2d_full(img, kernel, stride, pad, 96)
