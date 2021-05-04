#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import shutil
from os.path import isfile, join
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


# Uploading Video to desired path
video = cv2.VideoCapture('/notebooks/Notebooks_2019/Style_Data/Test4.mp4')

#os.path.exists('/notebooks/Notebooks_2019/Style_Data/Images_from_video')

# Deleting folder ( Images_from_video ) along with data present inside
try :
    if os.path.exists('/notebooks/Notebooks_2019/Style_Data/Images_from_video') :
        shutil.rmtree('/notebooks/Notebooks_2019/Style_Data/Images_from_video')
except OSError:
    print('Error in deleting directory')

# Deleting folder ( Video_from_images ) along with data present inside
try :
    if os.path.exists('/notebooks/Notebooks_2019/Style_Data/Video_from_images') :
        shutil.rmtree('/notebooks/Notebooks_2019/Style_Data/Video_from_images')
except OSError:
    print('Error in deleting directory')

# Creating folder ( Images_from_video )
try :
    if not os.path.exists('/notebooks/Notebooks_2019/Style_Data/Images_from_video') :
        os.makedirs('/notebooks/Notebooks_2019/Style_Data/Images_from_video')
except OSError:
    print('Error in creating directory')

# Creating folder ( Video_from_images )
try :
    if not os.path.exists('/notebooks/Notebooks_2019/Style_Data/Video_from_images') :
        os.makedirs('/notebooks/Notebooks_2019/Style_Data/Video_from_images')
except OSError:
    print('Error in creating directory')


# Function to extract image frames from video uploaded
def getFrame(sec) :
    # Seting current position of the video file in milliseconds.
    video.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    # Check if frame is read correctly
    video_length , image_frame = video.read()
    # Check if video_length is no more left to extract image frame
    if video_length :
        # Images name with which image frames will be saved
        image_name = '/notebooks/Notebooks_2019/Style_Data/Images_from_video/image_'+str(frame_count)+'.jpg'
        #print(image_name+' created')
        # Saving image files
        cv2.imwrite(image_name,image_frame)
    # returning whether video length is left or not to extract image frame
    return video_length

# Initalizing frame count to be used in naming image files
frame_count = 1
# Initalizing sec from where images will be extracted in video
sec = 0
# Initalizing frame rate ( frames per second ) with which images will be extracted
frame_Rate = 0.25
# Initalizing and checking if video length left after 1 iteriation
frame_check = getFrame(sec)
while frame_check :
    # Increasing frame count
    frame_count += 1
    # Increasing sec to sec + frame_Rate to move image extractin video forward 
    sec = sec + frame_Rate
    sec = round(sec,2)
    frame_check = getFrame(sec)
    
# Releasing video capture
video.release()
# Closing all frames
cv2.destroyAllWindows()

#frame_count = 0
#while(True) :
#    video_length , image_frame = video.read()
#    if video_length :
#        image_name = '/notebooks/Notebooks_2019/Style_Data/Images_from_video/'+str(frame_count)+'.jpg'
#        print(image_name+' created')
#        cv2.imwrite(image_name,image_frame)
#        frame_count += 1
#    else :
#        break
    
#video.release()
#cv2.destroyAllWindows()

#Test_image = Image.open('/notebooks/Notebooks_2019/Style_Data/Images_from_video/image_1.jpg')
#print(Test_image.size)
#(height,width) = Test_image.size
#New_height = 512
#height_percentage = New_height/height
#New_width = width*height_percentage
#print(New_height,int(New_width))

# Variable to check which device ( CPU or GPU - if available ) to use for processing
device_to_use = ('cuda' if torch.cuda.is_available() else 'cpu')

#print(device_to_use)

# Initalizing image size to convert all extracted images
imagesize = (512,512)

# Transformations to be applied to images before any processing
image_transformation = transforms.Compose([transforms.Resize(imagesize),transforms.ToTensor()])

#image = Image.open('/notebooks/Notebooks_2019/Style_Data/Test1.jpeg')
#print(image.size)
#print(image)
#image = image_transformation(image)
#print(image.shape)
#print(image)
#image = image.unsqueeze(0)
#print(image.shape)
#print(image)
#image = image.to(device_to_use, torch.float)
#print(image)

# Function to load images from defined path and load to device ( CPU / GPU )
def image_loader(path_to_image):
    # Image opening
    image = Image.open(path_to_image)
    # Transforming image
    image = image_transformation(image).unsqueeze(0)
    # Returing image after loading to device
    return image.to(device_to_use, torch.float)

#print(Actual_image.shape)
#print(Style_image.shape)

#unloader = transforms.ToPILImage()
#plt.figure()
#f, axis = plt.subplots(1,2)
#axis[0].imshow(unloader(Actual_image.cpu().clone().squeeze(0)))
#axis[1].imshow(unloader(Style_image.cpu().clone().squeeze(0)))

# Convert a tensor or an ndarray to PIL Image
unloader = transforms.ToPILImage()

# Function to display image
def image_display(tensor , title = None) :
    # Loading to CPU
    image = tensor.cpu().clone()
    # Remove single-dimensional entries from the shape.
    image = image.squeeze(0)
    # Converting to PIL Image
    image = unloader(image)
    # Displaying Image
    plt.imshow(image)
    if title is not None :
        # Setting image title
        plt.title(title)

# Function to calculate weighted version of the content distance for an individual layer
class PicContentLoss(nn.Module) :
    def __init__(self , target) :
        super(PicContentLoss , self).__init__()
        # Gradient is no longer being computed using detach()
        self.target = target.detach()
        
    def forward(self , input) :
        # calculating mean squared loss between input and target
        self.loss = F.mse_loss(input , self.target)
        # Returns the layer’s input
        return input

# Function to compute gram matrix used for calculating style loss
def gram_matrix(input) :
    batch_size , no_of_channels , height , width = input.size()
    # Resizing
    features_matrix = input.view(batch_size * no_of_channels , width * height)
    # Calculating gram matrix by multiplying feature matrix with its tranpose
    gram_product = torch.mm(features_matrix , features_matrix.t())
    # return normalized form of gram matrix to avoid any high value
    return gram_product.div(batch_size * no_of_channels * width * height)

# Function to calculate weighted version of the style distance for an individual layer
class PicStyleLoss(nn.Module) :
    def __init__(self , target_feature) :
        super(PicStyleLoss , self).__init__()
        # Gradient is no longer being computed using detach()
        self.target = gram_matrix(target_feature).detach()
        
    def forward(self , input) :
        Gram = gram_matrix(input)
        # calculating mean squared loss between gram matrix and target
        self.loss = F.mse_loss(Gram , self.target)
        return input

#target = cnn(Actual_image).detach()
#PicContentLoss(target).loss

# Importing VGG19 along with pretrained weights and set to eval mode
cnn = models.vgg19(pretrained=True).features.to(device_to_use).eval()
#print(cnn)
#for layer in cnn.children() : print(layer)

# Imagenet pretrianed model mean and standard deviation 
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device_to_use)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device_to_use)

# Function to normalize to input image
class Normalization_input(nn.Module) :
    def __init__(self,mean,std) :
        super(Normalization_input, self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)
        
    def forward(self , image_to_be_normalized) :
        return (image_to_be_normalized - self.mean) / self.std

# Desired layers to compute style/content losses
content_layers_default = ['conv_4']
style_layers_default = ['conv_1','conv_2','conv_3','conv_4','conv_5']

# Function to calulate loss as per layers
def get_style_model_and_losses(cnn ,normalization_mean ,normalization_std ,style_image ,
                               content_image ,content_layers = content_layers_default ,
                               style_layers = style_layers_default) :
    # Copy model
    cnn = copy.deepcopy(cnn)
    # Normalize input
    normalization_module = Normalization_input(normalization_mean,normalization_std).to(device_to_use)
    # Initalize both losses list to store
    content_losses = []
    style_losses = []
    # Initalize model and add layers in sequential order
    model = nn.Sequential(normalization_module)
    # Initalize layer number
    conv_layer = 0
    # Loop to add layer and layer name to model
    for layer in cnn.children() :
        if isinstance(layer ,nn.Conv2d) :
            conv_layer += 1
            layer_name = 'conv_{}'.format(conv_layer)
        elif isinstance(layer ,nn.ReLU) :
            layer_name = 'relu_{}'.format(conv_layer)
            layer = nn.ReLU(inplace = False)
        elif isinstance(layer ,nn.MaxPool2d) :
            layer_name = 'maxPool_{}'.format(conv_layer)
        elif isinstance(layer ,nn.BatchNorm2d) :
            layer_name = 'batchNorm_{}'.format(conv_layer)
        else :
            raise RuntimeError('Unrecognized layer : {}'.format(layer.__class__.__name__))
            
        model.add_module(layer_name , layer)
        #print(model)
        
        if layer_name in content_layers :
            # Add content loss 
            target = model(content_image).detach()
            #print(model(content_image).shape)
            #print(target.shape)
            Pic_ContentLoss = PicContentLoss(target)
            model.add_module('content_loss_{}'.format(conv_layer),Pic_ContentLoss)
            #print('content_loss_{}'.format(conv_layer))
            content_losses.append(Pic_ContentLoss)
        
        if layer_name in style_layers :
            # Add style loss
            target_feature = model(style_image).detach()
            #print(target_feature.shape)
            Pic_StyleLoss = PicStyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(conv_layer),Pic_StyleLoss)
            #print('style_loss_{}'.format(conv_layer))
            style_losses.append(Pic_StyleLoss)
            #print(style_losses)
         
        #print(conv_layer)
    # Loop from reverse to count no of layer in model excluding any flatten layer
    for conv_layer in range(len(model)-1,-1,-1) :
        if isinstance(model[conv_layer], PicContentLoss) or isinstance(model[conv_layer], PicStyleLoss) :
            break
    
    #print(conv_layer)  
    #print(model[conv_layer])
    # Final model over which training will be done
    model = model[:(conv_layer+1)]
    #print(model)
        
    return model , style_losses , content_losses

# Function defining optimizer
def input_optimizer (input_image) :
    optimizer = optim.LBFGS([input_image.requires_grad_()])
    return optimizer

# Function to execute neural transfer over image
def neural_transfer_execute (cnn ,normalization_mean ,normalization_std ,style_image ,content_image ,
                             input_image, num_of_steps = 150, style_weight = 100000, content_weight = 1) :
    
    print('Starting Neural Transfer Model')
    model , style_losses , content_losses = get_style_model_and_losses(cnn ,normalization_mean ,normalization_std ,
                                                                       style_image ,content_image)
    optimizer = input_optimizer (input_image)
    
    #print(model)
    
    print('Optimizing')
    run = [0]
    while run[0] <= num_of_steps :
        def input_image_update() :
            # Clamp all elements in input into the range [ min, max ] ; y = min(max(x,min_value),max_value)
            input_image.data.clamp_(0,1)
            # Sets gradients of all model parameters to zero.
            optimizer.zero_grad()
            model(input_image)
            style_score = 0
            content_score = 0
            #print(model)
            
            for sl in style_losses :
                #print('Entered Style')
                style_score += sl.loss
                #print(sl.loss)
            
            for cl in content_losses :
                content_score += cl.loss
                
            style_score *= style_weight
            content_score *= content_weight
            
            loss = style_score + content_score
            # Computes the derivative of the loss w.r.t. the parameters
            loss.backward()
            
            run[0] += 1
            
            if run[0] % 50 == 0 :
                print('Run {} : '.format(run))
                print('Style Loss : {:4f} Content Loss : {:4f}'.format(style_score.item() , content_score.item()))
                print()
                
            return style_score + content_score
        
        # Optimizer to take a step based on the gradients of the parameters
        optimizer.step(input_image_update)
        
    input_image.data.clamp_(0,1)
    # Return converted image
    return input_image

#unloader(output.cpu().clone().squeeze(0)).save('/notebooks/Notebooks_2019/Style_Data/Output.jpg')


path_modified_output_image = '/notebooks/Notebooks_2019/Style_Data/Video_from_images/'
path_input_image = '/notebooks/Notebooks_2019/Style_Data/Images_from_video/'

Style_image = image_loader('/notebooks/Notebooks_2019/Style_Data/Style15.jpg')

# Loop over all extracted images from video and convert them in new style
image_files = [file for file in os.listdir(path_input_image) if isfile(join(path_input_image,file))]
for x in range(len(image_files)) :
    input_image_filename = path_input_image+image_files[x]
    output_image_filename = path_modified_output_image+'Modified_'+image_files[x]
    #print(image_filename+' '+output_image_filename)
    image_size_check = Image.open(input_image_filename)
    (height,width) = image_size_check.size
    New_height = 512
    height_percentage = New_height/height
    # Calculating width as per height reduction percentage
    New_width = width*height_percentage
    print(height,width)
    print(New_height,int(New_width))
    imagesize = (New_height,int(New_width))
    Actual_image = image_loader(input_image_filename)
    input_image = Actual_image.clone()
    plt.figure()
    image_display(input_image,'Input Image')
    # Calling model to convert image
    output_image = neural_transfer_execute (cnn ,cnn_normalization_mean ,cnn_normalization_std ,Style_image ,Actual_image , input_image)
    plt.figure()
    image_display(output_image,'Output Image')
    output_image = output_image.cpu().clone().squeeze(0)
    output_image_transformation = transforms.Compose([transforms.Resize((height,width))])
    #output_image = output_image_transformation(output_image)
    print(output_image.shape)
    # Image save
    unloader(output_image).save(output_image_filename)

# Converting all Images to GIF
path_input_image = '/notebooks/Notebooks_2019/Style_Data/Video_from_images/'
path_output_image = '/notebooks/Notebooks_2019/Style_Data/final_video.mp4'
frame_Rate = 2
frame_array = []
image_files = [file for file in os.listdir(path_input_image) if isfile(join(path_input_image,file))]
#print(image_files)
image_files.sort(key=lambda x:int(x[15:-4]))
#print(image_files)
for x in range(len(image_files)) :
    image_filename = path_input_image+image_files[x]
    #print(image_filename)
    image = cv2.imread(image_filename)
    h,w,layers = image.shape
    size=(w,h)
    frame_array.append(image) 

    
output = cv2.VideoWriter(path_output_image,cv2.VideoWriter_fourcc(*'MJPG'),frame_Rate,size)
for i in range(len(frame_array)) :
    output.write(frame_array[i])
    
output.release()
video.release()
cv2.destroyAllWindows()
