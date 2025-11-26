
import torch
import torch.nn as nn
import math


class DeepQ_Model(nn.Module):
    def __init__(self, json_data):
        super().__init__()

        self.conv = nn.Sequential()

        m = json_data
        board_size = m["board_size"]
        frames = m["frames"]

        #Keep track of size of latest layer output
        dimensions = {
            'height': board_size,
            'width': board_size,
            'channels': frames
        }
        for layer in m['model']:
            l = m['model'][layer]
            #Add layer according to what type the json file indicates
            if('Conv2D' in layer):
                padding = 0
                if 'padding' in l.keys():
                    if l['padding'] == "same":
                        #Calculate padding to keep output size same as input size
                        #padding = 1
                        print((l['kernel_size'][0]-1)/2)
                        padding = int((l['kernel_size'][0]-1)/2)

                self.conv.append(
                    module=nn.Conv2d(
                    in_channels = dimensions['channels'], 
                    out_channels = l['filters'],
                    kernel_size = l['kernel_size'],
                    padding = padding
                    ))
                
                #Update dimensions
                dimensions['height'] = math.floor(dimensions['height']+2*padding-l['kernel_size'][1]) + 1
                dimensions['width'] = math.floor(dimensions['width']+2*padding-l['kernel_size'][0]) + 1
                dimensions['channels'] = l['filters']

                #Add activation function
                self.conv.append(nn.ReLU())
            if('Flatten' in layer):
                self.conv.append(nn.Flatten())
            if('Dense' in layer):
                self.conv.append(nn.Linear(
                    in_features=dimensions['height']*dimensions['width']*dimensions['channels'],
                    out_features=l['units']
                ))
                self.conv.append(nn.ReLU())

        self.out = nn.Linear(in_features=64, out_features=m['n_actions'])
        


    def forward(self, x: torch.Tensor):
        #print(x.shape)

        x = self.conv(x)
        #print(x.shape)

        x = self.out(x)
        #print(x.shape)

        return x
    