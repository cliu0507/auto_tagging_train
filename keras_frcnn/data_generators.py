from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools

def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height


class SampleSelector:
	def __init__(self, class_count):
		# ignore classes that have zero samples
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes)
		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):

	downscale = float(C.rpn_stride) # downscale = 16 (set in config)
	anchor_sizes = C.anchor_box_scales # anchor_sizes = 3
	anchor_ratios = C.anchor_box_ratios # anchor_ratios = 3
	num_anchors = len(anchor_sizes) * len(anchor_ratios) # num_anchors = 9
	# calculate the output map size based on the network architecture

	(output_width, output_height) = img_length_calc_function(resized_width, resized_height) #- featuremap_height/width
	#Assume resized_height = 1000, resized_weight = 600, use vgg 16
	#(output_width, output_height) = (37, 62)

	n_anchratios = len(anchor_ratios) # n_anchratios = 3
	
	# initialise empty output objectives
	#In np array , height first
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors)) # y_rpn_overlap.shape = [62,37,9] - means if this is a valid pos/neg anchor sample, if 1 means yes(ground truth mask)
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors)) # y_is_box_valid.shape = [62,37,9] - means positive or negetive anchor sample
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4)) # y_rpn_regr.shape = [62,37,9*4]

	num_bboxes = len(img_data['bboxes']) # number of bounding box in single groundtruth image

	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int) # num of anchor(positive) for each bounding box
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int) 
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32) # record the best iou so far in search
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

	# get the GT box coordinates, and resize to account for image resizing
	gta = np.zeros((num_bboxes, 4))
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
		gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
	
	# rpn ground truth
	i = 0
	
	for anchor_size_idx in range(len(anchor_sizes)): # anchor_sizes = 3
		for anchor_ratio_idx in range(n_anchratios): # n_anchratios = 3
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0] # anchor_x = 128 * 1 = 128
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1] # anchor_y = 128 * 1 = 128
			
			for ix in range(output_width):	 # output_width = 62			
				# x-coordinates of the current anchor box	
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2 # x1_anc = 16 * (0 + 0.5) - 128/2
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2 # x2_anc = 16 * (0 + 0.5) + 128/2
				
				# ignore boxes that go across image boundaries					
				if x1_anc < 0 or x2_anc > resized_width:
					continue
					
				for jy in range(output_height):

					# y-coordinates of the current anchor box
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y2_anc > resized_height:
						continue

					# bbox_type indicates whether an anchor should be a target 
					bbox_type = 'neg'
					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
					best_iou_for_loc = 0.0

					for bbox_num in range(num_bboxes):
						
						# get IOU of the current GT box and the current anchor box
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						# calculate the regression targets if they will be needed
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
							cxa = (x1_anc + x2_anc)/2.0
							cya = (y1_anc + y2_anc)/2.0

							tx = (cx - cxa) / (x2_anc - x1_anc)
							ty = (cy - cya) / (y2_anc - y1_anc)
							tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
						
						if img_data['bboxes'][bbox_num]['class'] != 'bg':

							# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
							if curr_iou > best_iou_for_bbox[bbox_num]:
								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
								best_iou_for_bbox[bbox_num] = curr_iou
								best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
								best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

							# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
							if curr_iou > C.rpn_max_overlap:
								bbox_type = 'pos'
								num_anchors_for_bbox[bbox_num] += 1
								# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
								if curr_iou > best_iou_for_loc:
									best_iou_for_loc = curr_iou
									best_regr = (tx, ty, tw, th)

							# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
								# gray zone between neg and pos
								if bbox_type != 'pos':
									bbox_type = 'neutral'

					# turn on or off outputs depending on IOUs
					if bbox_type == 'neg':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'neutral':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':
						i +=1
						#print((int(x1_anc), int(y1_anc)), (int(x2_anc), int(y2_anc)))
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
						y_rpn_regr[jy, ix, start:start+4] = best_regr

	# we ensure that every bbox has at least one positive RPN region

	for idx in range(num_anchors_for_bbox.shape[0]): # num_anchors_for_bbox.shape = [number of bbox,]
		if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1: # num_anchors_for_bbox[idx] = [30,20,0,1] , 0 means the first anchor ratio, 1 means second anchor size
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1)) # y_rpn_overlap.shape = [9,62,37] - [number of anchors, height, weight]
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0) # y_rpn_overlap.shape = [1,9,62,37] -  [1, number of anchors, height, weight]

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0) # y_is_box_valid.shape = [1,9,62,37]

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0) # y_is_box_valid.shape = [1,36,62,37]

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1)) #? pos_locs = (array(index1,,index2,..), array(index1,,index2,..),array(index1,,index2,..))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
	#pos_locs has three elements, first one store the indices of first index list of element y_rpn_overlap[0], 
	#Second one stores the indices of second index list of element y_rpn_overlap[0]
	#check https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
	#So one pos sample would be y_rpn_overlap[pos_locs[0][0],pos_locs[1][0],pos_locs[2][0]]

	num_pos = len(pos_locs[0])
	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.
	num_regions = 256

	global visualise
	visualise = False
	
	'''
	print("anchor_sizes:", anchor_sizes)
	print("anchor_ratios:", anchor_ratios)
	print("num_anchors",num_anchors)
	print("filename: ",img_data["filepath"])
	print("raw_width,raw_height:", (width, height))
	print("featuremap_width,featuremap_height: " ,(output_width, output_height))
	print("resized_width,resized_height: ", (resized_width, resized_height))
	print("positive anchor in preliminary search: ", i)
	print("final positive anchor point number:(anchor_id, height, width) ",len(pos_locs[0]))
	#print(pos_locs)
	'''


	#This section is to visualize the positive anchor
	if visualise:
		img = cv2.imread(img_data['filepath'])
		x_img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
		#print(x_img.shape)
		
		for bbox in img_data['bboxes']:
			gta_x1 = int(bbox['x1'] * (resized_width / float(width)))
			gta_x2 = int(bbox['x2'] * (resized_width / float(width)))
			gta_y1 = int(bbox['y1'] * (resized_height / float(height)))
			gta_y2= int(bbox['y2'] * (resized_height / float(height)))
			#print((gta_x1,gta_y1),(gta_x2,gta_y2))
			cv2.rectangle(x_img, (gta_x1,gta_y1),(gta_x2,gta_y2), (0, 0, 255),thickness=2)
		
		
		for pos_loc,ix,jy in zip(pos_locs[0],pos_locs[2],pos_locs[1]):
			#print("anchor_id is :",pos_loc)
			
			anchor_size_idx = pos_loc // len(anchor_ratios)
			anchor_ratio_idx = pos_loc % len(anchor_ratios)

			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]  # anchor_width
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]  # anchor_height
			#print("anchor_width, anchor_height: " ,(anchor_x,anchor_y))
			
			x1_anc =  int(downscale * (ix + 0.5) - anchor_x / 2)  # x1_anc = 16 * (0 + 0.5) - 128/2
			x2_anc =  int(downscale * (ix + 0.5) + anchor_x / 2)  # x2_anc = 16 * (0 + 0.5) + 128/2
			y1_anc =  int(downscale * (jy + 0.5) - anchor_y / 2)
			y2_anc =  int(downscale * (jy + 0.5) + anchor_y / 2)
			#print("Valid Positive:",(x1_anc,y1_anc),(x2_anc,y2_anc))
			cv2.rectangle(x_img, (x1_anc,y1_anc),(x2_anc,y2_anc), (255, 255, 0),thickness=2)
		
		x_img = cv2.resize(x_img, (width * 2 , height * 2), interpolation=cv2.INTER_CUBIC)
		cv2.imshow('img', x_img)
		cv2.waitKey(0)
	
	#print("\n")

	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), int(len(pos_locs[0]) - num_regions/2))
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), int(len(neg_locs[0]) - num_pos))
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0
	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1) # y_rpn_cls.shape = [1,9+9,62,37]
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1) # y_rpn_regr.shape = [1,9*4 + 36,62,37]

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

	
def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):

	# The following line is not useful with Python 3.5, it is kept for the legacy
	# all_img_data = sorted(all_img_data)

	sample_selector = SampleSelector(class_count)
	while True:
		if mode == 'train':
			np.random.shuffle(all_img_data)
		for img_data in all_img_data:
			try:
				if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
					continue
				# read in image, and optionally add augmentation
				if mode == 'train':
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
				else:
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)
				(width, height) = (img_data_aug['width'], img_data_aug['height'])
				(rows, cols, _) = x_img.shape
				assert cols == width
				assert rows == height
				# get image dimensions for resizing
				(resized_width, resized_height) = get_new_img_size(width, height, C.im_size)
				# resize the image so that smalles side is length = 600px
				x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
				try:
					#y_rpn_cls.shape = [1,9+9,62,37] - [1,number of anchors, featuremap_height, featuremap_width]
					#y_rpn_regr = [1,36+36,62,37]
					y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
				except:
					continue
				
				# Zero-center by mean pixel, and preprocess image
				# assume x_img.shape = [1000,600,[B,G,R]]
				x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB # x_img.shape = [1000,600,[R,G,B]]
				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]
				x_img /= C.img_scaling_factor
				x_img = np.transpose(x_img, (2, 0, 1)) # Change tp x_img.shape = [[R,G,B],1000,600]
				x_img = np.expand_dims(x_img, axis=0) # x_img.shape = [1,[R,G,B],1000,600]
				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling
				#y_rpn_regr[:, 36:, :, :] *= C.std_scaling
				
				if backend == 'tf':
					x_img = np.transpose(x_img, (0, 2, 3, 1)) # x_img.shape = [1,1000,600,[R,G,B]]
					y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1)) # y_rpn_cls.shape = [1,62,37,18]
					y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1)) # y_rpn_regr.shape = [1,62,37,72]
				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug # img_data_aug is [bounding box list]
			
			except Exception as e:
				print(e)
				continue
