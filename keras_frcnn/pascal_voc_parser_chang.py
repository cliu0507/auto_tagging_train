#This is the pascal_voc_parser to parse pascal voc format dataset

import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import uuid


#Chang's version on Banner Reflow Web Ads dataset
def get_data(input_path):
	'''function to read the path to folder storing xml files and get data
	args:
		input_path: string
			absolute path to the folder to store xml files
	returns:
		None
	'''
	all_imgs = [] #list to store all image paths and annotation data
	classes_count = {}
	class_mapping = {}
	data_path = input_path 
	visualise = False  
	annot_path = os.path.join(data_path,"XML_Chang") #annotation folder path
	imgs_path = os.path.join(data_path,"JPG_Chang") #raw image folder path
	#find all annotation xmls from annot_path
	annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]  
	idx = 0
	for annot in annots:
		try:
			idx += 1
			et = ET.parse(annot)
			element = et.getroot()
			element_filename = element.find('filename').text
			element_width = int(element.find('size').find('width').text)
			element_height = int(element.find('size').find('height').text)
			element_objs = element.findall('object')
			if len(element_objs) > 0:
				annotation_data = {'filepath': os.path.join(imgs_path, element_filename),'filename':element_filename, 'width': element_width,
									   'height': element_height, 'bboxes': []}
				
				#Init if_train variable to mark if this bbox is a training or test
				#if we enable train_test_split
				if_train = np.random.choice(np.arange(0, 2), p=[0.34,0.66])
				if if_train == 0:
					imageset = "test"
				else:
					imageset = "trainval"
				annotation_data['imageset'] = imageset

			for element_obj in element_objs:
				class_name = element_obj.find('name').text
				
				#This is to remove trailing number in class name such as "Headline1","Logo2"
				class_name = class_name.rstrip('1234567890.')
				
				# Merge some cases
				if class_name == "Offer":
					class_name = "Headline"
				if class_name == "Disclaimer":
					continue

				if class_name not in classes_count:
					classes_count[class_name] = 1
				else:
					classes_count[class_name] += 1
				if class_name not in class_mapping:
					class_mapping[class_name] = len(class_mapping)
				obj_bbox = element_obj.find('bndbox')
				x1 = int(round(float(obj_bbox.find('xmin').text)))
				y1 = int(round(float(obj_bbox.find('ymin').text)))
				x2 = int(round(float(obj_bbox.find('xmax').text)))
				y2 = int(round(float(obj_bbox.find('ymax').text)))
				difficulty = int(element_obj.find('difficult').text) == 1
				annotation_data['bboxes'].append(
						{	'class': class_name, 
							'x1': x1, 
							'x2': x2, 
							'y1': y1, 
							'y2': y2, 
							'difficult': difficulty
							})       
			all_imgs.append(annotation_data)
			if visualise:
				img = cv2.imread(annotation_data['filepath'])
				for bbox in annotation_data['bboxes']:
					cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
								  'x2'], bbox['y2']), (0, 0, 255))
				cv2.imshow('img', img)
				cv2.waitKey(0)
		except Exception as e:
			print(e)
			continue
	return all_imgs, classes_count, class_mapping

