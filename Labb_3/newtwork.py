
import os
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def network_setup(num_classes):
	model = models.Sequential()
	model.add( layers.Conv2D( 32, (3,3), activation = 'relu',
		input_shape = (150,150,3) ) )

	model.add( layers.MaxPooling2D( (2,2,) ) )

	model.add( layers.Conv2D( 64, (3,3), activation = 'relu'))
	model.add( layers.MaxPooling2D( (2,2,)) )
	model.add( layers.Conv2D( 128, (3,3), activation = 'relu'))
	model.add( layers.MaxPooling2D( (2,2,)) )
	model.add( layers.Conv2D( 128, (3,3), activation = 'relu'))
	model.add( layers.MaxPooling2D( (2,2,)) )

	model.add( layers.Flatten())
	model.add(layers.Dense(512, activation = 'relu'))
	model.add( layers.Dense( num_classes, activation = 'softmax'))
	
	model.compile( loss = 'categorical_crossentropy',
		optimizer = optimizers.RMSprop( lr = 1e-4),
		metrics =['acc'])
	return model

def plot_acc_loss(history ):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = history.epoch
	plt.plot(epochs, acc, 'bo', label = 'Train acc')
	plt.plot(epochs, val_acc, 'b', label = 'val acc')
	plt.title("Train and validation accuracy")
	plt.legend()
	plt.figure()

	plt.plot(epochs, loss, 'bo', label = 'Train loss')
	plt.plot(epochs, val_loss, 'b', label = 'val loss')
	plt.title("Train and validation loss")
	plt.legend()

	plt.show()



def preprocess_data():
	num_classes = 5

	class_dir = "flower_split"
	train_dir = "train"
	val_dir = "validation"
	test_dir = "test"

	train_datagen = ImageDataGenerator(rescale=1. / 255)
	val_datagen = ImageDataGenerator(rescale=1. / 255)
	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
		os.path.join(class_dir, train_dir),
		target_size = (150,150),
		batch_size = 20,
		class_mode = 'categorical')

	validation_generator = val_datagen.flow_from_directory(
		os.path.join(class_dir, val_dir),
		target_size = (150,150),
		batch_size = 20,
		class_mode = 'categorical')

	test_generator = test_datagen.flow_from_directory(
		os.path.join(class_dir, test_dir),
		target_size = (150,150),
		batch_size = 20,
		class_mode = 'categorical')

	return train_generator, validation_generator, test_generator	

def test_prediction( model, test_generator):
	Y_pred = model.predict_generator(test_generator)
	y_pred = np.argmax(Y_pred, axis=1)
	print('Confusion Matrix')
	print(confusion_matrix(test_generator.classes, y_pred))
	print('Classification Report')
	target_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', "Tulip"]
	print(classification_report(test_generator.classes, y_pred, target_names=target_names))


	return Y_pred, y_pred


def data_augmentation():

	datagen = ImageDataGenerator(
		rotation_range = 40,
		width_shift_range = 0.2,
		height_shift_range = 0.2,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True,
		fill_mode = 'nearest')


def show_augmented_flower():
	datagen = ImageDataGenerator(
		rotation_range = 40,
		width_shift_range = 0.2,
		height_shift_range = 0.2,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True,
		fill_mode = 'nearest')
	class_dir = "flowers"
	pic_paths = [os.path.join(class_dir,'tulip', pic) for 
	pic in os.listdir(class_dir +'\\tulip')]

	pic_path = pic_paths[3]
	pic = image.load_img( pic_path, target_size = (150,150))

	x = image.img_to_array(pic)
	x = x.reshape((1,) + x.shape)
	i=0
	for batch in datagen.flow(x, batch_size=1):
		plt.figure(i)
		imgplot = plt.imshow(image.array_to_img(batch[0]))
		i += 1
		if i % 4 == 0:
			break
	plt.show()


if __name__ == '__main__':
	show_augmented_flower()
	'''
	train_generator, validation_generator, test_generator = preprocess_data()
	"""
	model = network_setup(num_classes)

	history = model.fit_generator(
		train_generator,
		steps_per_epoch = 50,
		epochs = 10,
		validation_data = validation_generator,
		validation_steps = 25)


	model.save('flower_model.h5')
	plot_acc_loss(history)
	"""
	model = models.load_model('flower_model.h5')



	y1, y2 = test_prediction(model, test_generator)
	'''










