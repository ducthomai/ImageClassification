from keras.preprocessing.image import ImageDataGenerator, img_to_array, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range= 40,
                             width_shift_range= 0.2,
                             height_shift_range= 0.2,
                             shear_range= 0.2,
                             zoom_range= 0.2,
                             horizontal_flip= True,
                             fill_mode= 'nearest')
img = load_img('Identify/Lactobacillus.johnsonii/Lactobacillus.johnsonii_0020.tif')
x = img_to_array(img)
x = x.reshape((1,)+x.shape)

i=0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='Identify/Lactobacillus.johnsonii',
                          save_prefix= 'Lactobacillus.johnsonii_0020',
                          save_format='png'):
    i+= 1
    if i > 8:
        break


