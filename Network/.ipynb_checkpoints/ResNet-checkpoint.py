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
    
    def __init__(self, layer_dims, num_classes=5):
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
    

class FBNet(Model):
    """
    LeNet is an early and famous CNN architecture for image classfication task.
    It is proposed by Yann LeCun. Here we use its architecture as the startpoint
    for your CNN practice. Its architecture is as follow.

    input >> Conv2DLayer >> Conv2DLayer >> flatten >>
    DenseLayer >> AffineLayer >> softmax loss >> output

    Or

    input >> [conv2d-avgpooling] >> [conv2d-avgpooling] >> flatten >>
    DenseLayer >> AffineLayer >> softmax loss >> output

    http://deeplearning.net/tutorial/lenet.html
    """

    def __init__(self, input_shape, output_size=10):
        '''
        input_shape: The size of the input. (img_len, img_len, channel_num).
        output_size: The size of the output. It should be equal to the number of classes.
        '''
        super(FBNet, self).__init__()
        #############################################################
        # TODO: Define layers for your custom LeNet network         
        # Hint: Try adding additional convolution and avgpool layers
        #############################################################
        
        self.conv_layer_1 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1,1), activation='relu',padding='same',input_shape = input_shape)
        self.conv_layer_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), activation='relu',padding='same')
        
        self.maxpool_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding = 'same' )
        
        self.bn_layer = BatchNormalization()
        self.avgpool_layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.flatten_layer = Flatten()
        self.dropout = Dropout(0.5)
        
        self.fc_layer_1 = Dense(1024, activation='relu')
        self.fc_layer_2 = Dense(512, activation='relu')
        self.fc_layer_3 = Dense(output_size, activation='softmax')
        #############################################################
        #                          END TODO                         #                                              
        #############################################################
        
        
    def call(self, x):
        '''
        x: input to LeNet model.
        '''
        #call function returns forward pass output of the network
        #############################################################
        # TODO: Implement forward pass for custom network defined 
        # in __init__ and return network output
        #############################################################
        
        x = self.conv_layer_1(x)
        x = self.maxpool_layer(x)
        x = self.conv_layer_2(x)
        x = self.avgpool_layer(x)
        x = self.flatten_layer(x)
        x = self.fc_layer_1(x)
        x = self.dropout(x)
        x = self.fc_layer_2(x)
        x = self.dropout(x)
        out = self.fc_layer_3(x)
        
        return out
        #############################################################
        #                          END TODO                         #                                              
        #############################################################
        
        
class FBNet_T(Model):
    """
    LeNet is an early and famous CNN architecture for image classfication task.
    It is proposed by Yann LeCun. Here we use its architecture as the startpoint
    for your CNN practice. Its architecture is as follow.

    input >> Conv2DLayer >> Conv2DLayer >> flatten >>
    DenseLayer >> AffineLayer >> softmax loss >> output

    Or

    input >> [conv2d-avgpooling] >> [conv2d-avgpooling] >> flatten >>
    DenseLayer >> AffineLayer >> softmax loss >> output

    http://deeplearning.net/tutorial/lenet.html
    """

    def __init__(self, input_shape, output_size=10):
        '''
        input_shape: The size of the input. (img_len, img_len, channel_num).
        output_size: The size of the output. It should be equal to the number of classes.
        '''
        super(FBNet_T, self).__init__()
        #############################################################
        # TODO: Define layers for your custom LeNet network         
        # Hint: Try adding additional convolution and avgpool layers
        #############################################################
        
        
        
        self.conv_layer_1a = Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), activation='relu',padding='same',input_shape = input_shape)
        self.conv_layer_1b = Conv2D(filters=32, kernel_size=(5, 5), strides=(1,1), activation='relu',padding='same',input_shape = input_shape)
        self.conv_layer_1c = Conv2D(filters=32, kernel_size=(7, 7), strides=(1,1), activation='relu',padding='same',input_shape = input_shape)
        
        
        self.conv_layer_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), activation='relu',padding='same')
        
        self.maxpool_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding = 'same' )
        
        self.bn_layer = BatchNormalization()
        self.avgpool_layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.flatten_layer = Flatten()
        self.dropout = Dropout(0.5)
        
        self.fc_layer_1 = Dense(1024, activation='relu')
        self.fc_layer_2 = Dense(512, activation='relu')
        self.fc_layer_3 = Dense(output_size, activation='softmax')
        #############################################################
        #                          END TODO                         #                                              
        #############################################################
        
        
    def call(self, x):
        '''
        x: input to LeNet model.
        '''
        #call function returns forward pass output of the network
        #############################################################
        # TODO: Implement forward pass for custom network defined 
        # in __init__ and return network output
        #############################################################
        
        
        x1 = x; x2 = x; x3 = x
        
        
        x1 = self.conv_layer_1a(x1)
        x1 = self.maxpool_layer(x1)
        x1 = self.conv_layer_2(x1)
        x1 = self.avgpool_layer(x1)
        x1 = self.flatten_layer(x1)
        x1 = self.fc_layer_1(x1)
        x1 = self.dropout(x1)
        x1 = self.fc_layer_2(x1)
        
        x2 = self.conv_layer_1b(x2)
        x2 = self.maxpool_layer(x2)
        x2 = self.conv_layer_2(x2)
        x2 = self.avgpool_layer(x2)
        x2 = self.flatten_layer(x2)
        x2 = self.fc_layer_1(x2)
        x2 = self.dropout(x2)
        x2 = self.fc_layer_2(x2)
        
        x3 = self.conv_layer_1c(x3)
        x3 = self.maxpool_layer(x3)
        x3 = self.conv_layer_2(x3)
        x3 = self.avgpool_layer(x3)
        x3 = self.flatten_layer(x3)
        x3 = self.fc_layer_1(x3)
        x3 = self.dropout(x3)
        x3 = self.fc_layer_2(x3)
        
        x = tf.concat([x1,x2,x3],1)
        
        
        out = self.fc_layer_3(x)
        
        return out
        #############################################################
        #                          END TODO                         #                                              
        #############################################################
        
        
