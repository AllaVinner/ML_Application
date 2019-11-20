from os import listdir
from random import shuffle
"""

"""
import os 
import shutil






if __name__ == '__main__':
	path_to_classes = '.\\flowers'

	# List all classes
	classes = listdir( path_to_classes)

	# Store all samples under their cathegory
	class_2_sample = dict()
	for class_ in classes:
		path_to_class = path_to_classes +"\\" + class_
		class_2_sample[class_] = listdir( path_to_class )
		
	# create a list of tuples (file_name, cathegory)
	data = []
	for k, v in class_2_sample.items():
		for sample_name in v:
			data.append( (sample_name, k))

	train_size = int(0.6*len(data))
	validation_size = int(0.2*len(data))
	test_size = len(data) - train_size - validation_size

	# Split the set randomly into the 3 cathegroies
	shuffle(data)
	train_images = data[0:train_size]
	val_images= data[train_size:train_size+validation_size]
	test_images = data[-test_size:]


	org_class_dir = "flowers"
	current_dir = "flower_split"
	train_dir = "train"
	val_dir = "validation"
	test_dir = "test"

	for image, label in train_images:
	    src = os.path.join(org_class_dir, label, image)
	    dst = os.path.join(current_dir, train_dir, label, image)
	    os.makedirs(os.path.dirname(dst), exist_ok=True)
	    shutil.copyfile(src, dst)

	for image, label in val_images:
	    src = os.path.join(org_class_dir,label, image)
	    dst = os.path.join(current_dir, val_dir, label, image)
	    os.makedirs(os.path.dirname(dst), exist_ok=True)
	    shutil.copyfile(src, dst)

	for image, label in test_images:
	    src = os.path.join(org_class_dir , label, image)
	    dst = os.path.join(current_dir, test_dir, label, image)
	    os.makedirs(os.path.dirname(dst), exist_ok=True)
	    shutil.copyfile(src, dst)







