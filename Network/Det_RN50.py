from tensorflow.keras import Model, applications
from tensorflow.keras.layers import  Dense,GlobalAveragePooling2D

def Det_RN50():
    base_model = applications.ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    # for layer in base_model.layers[:140]:  # Keep the pretrained params
    # 	   layer.trainable = True
    # for layer in base_model.layers[140:]:  # Keep the pretrained params
    # 	   layer.trainable = True
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, name='predictions')(x)
    model = Model(base_model.input,predictions)
    return model


