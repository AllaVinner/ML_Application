
import os
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pickle

from keras.applications import VGG16


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
	model.add(layers.Dropout(0.5))
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


def aug_preprocess_data():
	num_classes = 5

	class_dir = "flower_split"
	train_dir = "train"
	val_dir = "validation"
	test_dir = "test"

	train_dir = os.path.joinr(class_dir, train_dir)
	val_dir = os.path.joinr(class_dir, val_dir)
	test_dir = os.path.joinr(class_dir, test_dir)

	train_datagen = ImageDataGenerator(
		rescale=1./255,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,)

	test_datagen = ImageDataGenerator(rescale=1./255)
	train_generator = train_datagen.flow_from_directory(
		train_dir,
		target_size=(150, 150),
		batch_size=32,
		class_mode='categorical')
	validation_generator = test_datagen.flow_from_directory(
		val_dir,
		target_size=(150, 150),
		batch_size=32,
		class_mode='categorical')

	test_generator = test_datagen.flow_from_directory(
		test_dir,
		target_size=(150, 150),
		batch_size=32,
		class_mode='categorical')

	return train_generator,validation_generator, test_generator


def pre_preprocess_data():

	sample_count = 100
	batch_size = 20

	conv_base = VGG16(weights='imagenet',
		include_top=False,
		input_shape=(150, 150, 3))

	class_dir = "flower_split"
	train_dir = "train"
	val_dir = "validation"
	test_dir = "test"

	train_dir = os.path.join(class_dir, train_dir)
	val_dir = os.path.join(class_dir, val_dir)
	test_dir = os.path.join(class_dir, test_dir)

	train_features, train_labels = extract_features(conv_base, train_dir, 2000, batch_size)
	validation_features, validation_labels = extract_features(conv_base, val_dir, 1000, batch_size)
	test_features, test_labels = extract_features(conv_base, test_dir, 1000, batch_size)

	train_features = np.reshape(train_features, (2000, 4*4* 512))
	validation_features = np.reshape(validation_features, (1000, 4*4* 512))
	test_features = np.reshape(test_features, (1000, 4*4* 512))

	return train_features, train_labels, validation_features, validation_labels, test_features, test_labels


def pre_network_setup(num_classes):
	model = models.Sequential()
	model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense( num_classes , activation='softmax'))


	model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
		loss='categorical_crossentropy',
		metrics=['acc'])

	return model

def extract_features(conv_base,directory, sample_count, batch_size):
	features = np.zeros(shape=(sample_count, 4, 4, 512)) #(4,4,512) output of pretrained
	labels = np.zeros(shape=(sample_count,5))
	datagen = ImageDataGenerator(rescale=1./255)
	generator = datagen.flow_from_directory(
		directory,
		target_size=(150, 150),
		batch_size=batch_size,
		class_mode='categorical')
	i=0
	for inputs_batch, labels_batch in generator:
		features_batch = conv_base.predict(inputs_batch)
		features[i * batch_size : (i + 1) * batch_size] = features_batch
		labels[i * batch_size : (i + 1) * batch_size] = labels_batch
		i += 1
		if i % 100 == 0:
			print("i is: {}\nSample count: {}\n".format(i, sample_count))
		if i * batch_size >= sample_count:
			break
	return features, labels

if __name__ == '__main__':

	"""
	# Train network
	train_generator, validation_generator, test_generator = preprocess_data()
	
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


	"""
	# Test network on test data and get accuracy and confusion matrix...
	model = models.load_model('flower_model.h5')
	y1, y2 = test_prediction(model, test_generator)
	"""


	###########################################################################



	"""
	# Shows distorted images
	show_augmented_flower()	
	"""		

	"""	
    # Train with augmented picture
	train_generator, validation_generator, test_generator = preprocess_data()
	
	model = network_setup(num_classes)

	history = model.fit_generator(
		train_generator,
		steps_per_epoch=100,
		epochs=100,
		validation_data=validation_generator,
		validation_steps=50)
	
	model.save('flower_model_aug.h5')
	plot_acc_loss(history)
	"""

	"""
	# Test network on test data and get accuracy and confusion matrix...
	model = models.load_model('flower_model_aug.h5')
	y1, y2 = test_prediction(model, test_generator)
	"""	

	###########################################################################

	
	# Extract features
	num_classes = 5
	train_features, train_labels, validation_features, validation_labels, test_features, test_labels = pre_preprocess_data()


	feat_map = {}
	feat_map['train_features'] = train_features
	feat_map['train_labels'] = train_labels
	feat_map['validation_features'] = validation_features
	feat_map['validation_labels'] = validation_labels
	feat_map['test_features'] = test_features
	feat_map['test_labels'] = test_labels

	with open('feat_map.p', 'wb') as fp:
		pickle.dump(feat_map,fp)


	"""
	# Train network
	num_classes = 5
	with open('feat_map.p', 'rb') as fp:
		feat_map = pickle.load(fp)

	train_features = feat_map['train_features']
	train_labels = feat_map['train_labels']
	validation_features = feat_map['validation_feature']
	validation_labels= feat_map['validation_labels']
	test_features = feat_map['test_features']
	test_labels = feat_map['test_labels']

	model = network_setup(num_classes)

	history = model.fit(train_features, train_labels,
		epochs=30,
		batch_size=20,
		validation_data=(validation_features, validation_labels))

	model.save('flower_model_pre.h5')

	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

	"""









