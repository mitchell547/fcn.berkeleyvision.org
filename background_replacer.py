# Video background replacer
import cv2
import numpy as np
import scipy.misc
import sys
import os
import argparse
os.environ['GLOG_minloglevel'] = '2' # disable a lot of Caffe output into the shell
import caffe
from PIL import Image

# Get mask of objects from source image using Caffe network (FCN)
# Source img is OpenCV-formatted
def get_mask(source, net):
	in_ = source.astype('float32')
	mean = np.array(cv2.mean(source), dtype=np.float32)
	in_ -= np.array(mean[0:3])
	in_ = in_.transpose((2,0,1))

	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_

	# run net and take argmax for prediction
	net.forward()
	out = net.blobs['score'].data[0].argmax(axis=0)
	return out

# Replace background from source img with new background using mask of objects
# All images must be in PIL-format (RGB)
# Black color of mask is transparent (background) color
def replace_background_pil(source, mask, background):
	source_data = source.getdata()
	mask_data = mask.getdata()
	background = background.resize((source.size), Image.ANTIALIAS)
	background_data = background.getdata()
	
	result_data = []
	for index, item in enumerate(mask_data):
		if item[0] == 0 and item[1] == 0 and item[2] == 0:
			result_data.append(background_data[index])
		else:
			result_data.append(source_data[index])
			
	result_img = Image.new("RGB", source.size)
    	result_img.putdata(result_data)
	return result_img
	
# Replace_background function for OpenCV-formatted images
# Source and Background images in OpenCV-format, Mask in FCN output format
def replace_background_opencv(source, mask, background):
	h, w, d = source.shape
	background = cv2.resize(background, (w, h))
	
	mask_fgd = np.where((mask==0),0,1).astype('uint8')
	mask_bgd = np.where((mask!=0),0,1).astype('uint8')
	
	fgd = source * mask_fgd[:,:,np.newaxis]
	src_bgd = source * mask_bgd[:,:,np.newaxis]
	bgd = background * mask_bgd[:,:,np.newaxis]
	
	alpha = 0.5 # how much background to add
	result_img = fgd + src_bgd * (1-alpha) + bgd * alpha
	result_img = result_img.astype('uint8')
	return result_img
	
# Some helpful conversions
def pil2opencv(pil_img):
	#pil_img = pil_img.convert("RGB")
	res = np.array(pil_img)
	res = res[:, :, ::-1].copy() # RGB to BGR
	return res

def opencv2pil(opencv_img):
	#opencv_img = cv.cvtColor(opencv_img, cv.COLOR_BGR2RGB)
	opencv_img = opencv_img[:, :, ::-1].copy() # BGR to RGB
	res = Image.fromarray(opencv_img)
	return res
	
# MAIN #
def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("source_video", help = "Source video to be processed (tested only .mp4 format)")
	parser.add_argument("background_image", help = "Background image that replaces original backgorund")
	parser.add_argument("--out", default = "result.avi", help = "Output video filename (tested only .avi format)")
	parser.add_argument("--codec", default = 'XVID', help = "FourCC of video codec to compress frames (default is XVID)")
	args = parser.parse_args()	
	
	# load FCN
	print("Loading FCN...")
	net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
	# gpu mode is highly recommended, because cpu is ~40 times slower
	caffe.set_device(0)
	caffe.set_mode_gpu() 
	
	# load resources
	source_vid = cv2.VideoCapture(args.source_video)
	fps = source_vid.get(cv2.cv.CV_CAP_PROP_FPS)
	print("%s; FPS: %f" % (args.source_video, fps))

	background_img = cv2.imread(args.background_image)
	#background_img = Image.open(args.background_image)
	
	frame_width = int(source_vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	frame_height = int(source_vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	result_vid = cv2.VideoWriter(args.out, cv2.cv.CV_FOURCC(*args.codec), fps, (frame_width, frame_height))

	# Main Loop #
	count = 0
	print("Replacing original background (may take a while)...")
	ret = True
	while(ret):
		ret, frame = source_vid.read()
		if not ret:
			break
			
		# For PIL images
		#frame_img = opencv2pil(frame)
		#mask = get_mask(frame_img, net)
		#mask_img = scipy.misc.toimage(out, cmin=0.0).convert("RGB")
		#result_img = replace_background_pil(frame_img, mask_img, background_img)
		#result_frame = pil2opencv(result_img)
		
		# For OpenCV images
		mask_img = get_mask(frame, net)
		result_frame = replace_background_opencv(frame, mask_img, background_img)
		
		result_vid.write(result_frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			
		count += 1

	if count > 0:	
		print("Background replaced. Processed: %d frames." % count)
		print("Result saved in: %s" % args.out)
	else:
		print("Something goes wrong. Could not read video frame")
	
	source_vid.release()
	result_vid.release()
	cv2.destroyAllWindows()
	os.environ['GLOG_minloglevel'] = '0'

if __name__ == '__main__':
	main(sys.argv)
