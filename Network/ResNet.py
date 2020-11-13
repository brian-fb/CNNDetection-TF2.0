import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,Input,GlobalAveragePooling2D,AveragePooling2D, MaxPooling2D
from tensorflow.keras import Model

class BasicBlock(layers.Layer):
    def __init__(self, num_filter, stride=1):
        super(BasicBlock, self).__init__()
        self.conv_layer_1 = layers.Conv2D(num_filter, (3,3), strides=stride, padding='same')
        self.conv_layer_2 = layers.Conv2D(num_filter, (3,3), strides=1, padding='same')
        
        self.relu = layers.Activation('relu')
        self.bn_layer = layers.BatchNormalization()
      
        if stride!=1:
            self.downsample=Sequential()
            self.downsample.add(layers.Conv2D(num_filter,(1,1),strides=stride))
        else:
            self.downsample=lambda x:x
            
    def call(self,x,training=None):
        
        x_ori = x
        
        x = self.conv_layer_1(x)
        x = self.bn_layer(x)
        x = self.relu(x)
        
        x = self.conv_layer_2(x)
        x = self.bn_layer(x)
        
        shortcut = self.downsample(x_ori)
        
        x = layers.add([x, shortcut])
        x = tf.nn.relu(x)
        
        return x
    
class ResNet(Model):
    
    def build_resblock(self,num_filter,blocks,stride=1):
        res_blocks= Sequential()
        # may down sample
        res_blocks.add(BasicBlock(num_filter,stride))
        # just down sample one time
        for pre in range(1,blocks):
            res_blocks.add(BasicBlock(num_filter,stride=1))
        return res_blocks
    
    def __init__(self, layer_dims, num_classes=2):
        super(ResNet, self).__init__()
        
        scale = 1
        
        self.pre_layer = Sequential([layers.Conv2D(int(64/scale),(3,3),strides=(1,1)),
                                    layers.BatchNormalization(),
                                    layers.Activation('relu'),
                                    layers.MaxPool2D(pool_size=(2,2), strides=(1,1),padding='same')])
        
        self.layer1 = self.build_resblock(int(64/scale), layer_dims[0])
        self.layer2 = self.build_resblock(int(128/scale), layer_dims[1], stride = 2)
        self.layer3 = self.build_resblock(int(256/scale), layer_dims[2], stride = 2)
        self.layer4 = self.build_resblock(int(512/scale), layer_dims[3], stride = 2)
        
        self.dropout = layers.Dropout(0.5)
        
        self.avgpool=layers.GlobalAveragePooling2D()
        self.fc1=layers.Dense(1024,activation='relu')
        self.fc2=layers.Dense(512,activation='relu')
        self.fc=layers.Dense(num_classes)
        
    def call(self,x,training=None):
        x=self.pre_layer(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        # [b,c]
        x=self.avgpool(x)
        x=self.fc1(x)
        x = self.dropout(x)
        x=self.fc2(x)
        x = self.dropout(x)
        x=self.fc(x)
        return x
    