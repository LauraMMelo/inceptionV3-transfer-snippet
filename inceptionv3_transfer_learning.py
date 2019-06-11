def inception(use_imagenet=True):
    # load pre-trained model graph, don't add final layer
    model = keras.applications.InceptionV3(include_top=False, input_shape=(299, 299, 3),
                                          weights='imagenet' if use_imagenet else None)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.GlobalAveragePooling2D()(model.output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(num_classes, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model


#Sumário
model = inception()
print(len(model.layers))

#trainable layers
# set all layers trainable by default
for layer in model.layers:
    layer.trainable = True
    if isinstance(layer, keras.layers.BatchNormalization):
        # we do aggressive exponential smoothing of batch norm
        # parameters to faster adjust to our new dataset
        layer.momentum = 0.9
    
# fix deep layers (fine-tuning only last 50)
for layer in model.layers[:-50]:
    # fix all but batch norm layers, because we neeed to update moving averages for a new dataset!
    if not isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

model.summary()

# Treinamento
adamax = keras.optimizers.Adamax(lr=0.01, 
                             beta_1=0.9, 
                             beta_2=0.999, 
                             epsilon=None, 
                             decay=0.0)

model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer= adamax,
                        metrics=['accuracy'])

#treinando o modelo com o novo dataset
model_train_dropout = model.fit_generator(generator=my_training_batch_generator,
                                              steps_per_epoch= num_training_samples // batch_size // 8,
                                              epochs=epochs,
                                              verbose=1,
                                              validation_data=my_validation_batch_generator,
                                              validation_steps= num_validation_samples // batch_size // 4)


#Salvando
model.save("car_brand_model_Inceptionv3.h5py")

