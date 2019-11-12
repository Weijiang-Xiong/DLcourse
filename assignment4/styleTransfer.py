# this file presents a style transfer example with pytorch.
#%%
from __future__ import print_function

import copy # to deep copy the models
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

#%% use GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imsize = 128 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize((imsize,imsize)),
    transforms.ToTensor()
])

def image_loader(iamge_path):
    image = Image.open(iamge_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_img = image_loader('./assignment4/images/style1.jpg')
content_img = image_loader('./assignment4/images/content.jpg')

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

#%%
unloader = transforms.ToPILImage()
plt.ion()

def imshow(tensor, title = None):
    image = tensor.cpu().clone()  # get a copy of the tensor
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title != None:
        plt.title(title)
    plt.pause(0.001)

plt.figure()
imshow(style_img, title = 'Style')

plt.figure()
imshow(content_img, title = 'Content')
print()


#%%
# the content loss is just the mean square error of the pixels
class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # Returns a new Tensor, detached from the current graph.
        # The result will never require gradient.
        self.target = target.detach()

    def forward(self, in_feat):
        self.loss = F.mse_loss(in_feat, self.target)
        return in_feat

#%%
# the style error depends on the gram matrix of features in a certain layer. 
# see lecture notes for details
def gram_matrix(in_feat):
    a, b, c, d = in_feat.size()
    # a: batch size, b: number of channels, c,d: dimensions of feature map.

    features = in_feat.view(a*b, c*d)

    gram = torch.mm(features, features.t()) 
    # the size of gram matrix is not related to the size of the feature map

    return gram.div(a * b * c * d)

class StyleLoss(nn.Module):
    
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, in_feat):
        gram = gram_matrix(in_feat)
        self.loss = F.mse_loss(gram, self.target)
        return in_feat

#%%
# os.environ['TORCH_HOME'] = 'assignment4'
cnn = models.vgg19_bn(pretrained=True).features.to(device).eval()
net_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
net_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # change the shape of mean and std to [C x 1 x 1],
        #  so that it can be broadcasted into input tensor [B x C x H x W]
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


#%%5
content_layers_default = ['conv_2']
style_layers_default = ['conv_2','conv_3']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


#%% 
unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

#%%
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

#%%
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


#%%
input_img = content_img.clone()
# input_img = torch.randn(content_img.data.size(), device=device)
content_img = image_loader('./assignment4/images/content.jpg')
style_img = image_loader('./assignment4/images/style4.jpg')

# add the original input image to the figure:
plt.figure()
imshow(style_img, title='Input Image')
output = run_style_transfer(cnn, net_mean,
                            net_std,
                            content_img,
                            style_img,
                            input_img,
                            num_steps=300,
                            style_weight=7777)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()

#%%
