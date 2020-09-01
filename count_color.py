import cv2
import numpy as np
import os
from pdf2image import convert_from_path, convert_from_bytes

def isItColor1(imgName):
	image = cv2.imread(os.path.join(directory, imgName))
	allSame = True
	image = np.array(image, dtype=np.int16)
	thresh = 15
	for x in range (0,image.shape[1]-1,1):
		if not allSame:
			break
		for y in range(0,image.shape[0]-1,1):
			R, G, B = image[y,x]
			if abs(R - G) > thresh or abs(R - B) > thresh or abs(G - B) > thresh:
				print(abs(R - G))
				print(abs(R - B))
				print(abs(G - B))
				allSame = False
				break
	return allSame

def isItColor2(imgName):
	image = cv2.imread(os.path.join(directory, imgName))
	count = 0
	for x in range (0,image.shape[1]-1,1):
		for y in range(0,image.shape[0]-1,1):
			R, G, B = image[y,x]
			if R != G or G != B or R != B:
				count = count + 1
				break
	return count * 100 / (image.shape[1] * image.shape[0])

def isItColor3(imgName):
	image = cv2.imread(os.path.join(directory, imgName))
	gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# cv2.imwrite("gimage.jpg", gimage)
	gimage = cv2.cvtColor(gimage, cv2.COLOR_GRAY2BGR)
	# cv2.imwrite("gimage 2.jpg", gimage)
	res = cv2.subtract(image, gimage)
	# cv2.imwrite(imgName + "res.jpg", res)
	return 100 * np.sum(res) / (3 * 255 * image.shape[1] * image.shape[0])

def isItColor4(imgName):
	image = cv2.imread(os.path.join(directory, imgName))
	gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gimage = cv2.cvtColor(gimage, cv2.COLOR_GRAY2BGR)
	res = cv2.subtract(image, gimage)
	final_img = cv2.bitwise_and(res, image)
	# cv2.imwrite(imgName + "fin.jpg", final_img)
	return 100 * np.sum(final_img) / (3 * 255 * image.shape[1] * image.shape[0])

def pdfToImages(pdfName, saveFolder):
	convert_from_path(pdfName, dpi=72, output_folder=saveFolder, first_page=None, last_page=None, fmt='jpeg',
	thread_count=4, userpw=None, use_cropbox=False, strict=False, transparent=False, single_file=False,
  	output_file=pdfName, poppler_path=".\\poppler-0.90.1\\bin", grayscale=False, size=None)

directory = ".\\temp"

# white = np.ones((100,200,3))
# white = white * 255
# print(np.sum(white))
# print((3 * 255 * white.shape[1] * white.shape[0]))
# exit(0)


if os.path.exists(directory):
	images = os.listdir(directory)
	for image in images:
		os.remove(os.path.join(directory, image))
	os.rmdir(directory)


os.makedirs(directory)
pdfToImages("srv.pdf",directory)
# 0.001 seems to be a good threshold
images = os.listdir(directory)
pageNo = 0
for image in images:
	pageNo = pageNo + 1
	cValue = isItColor4(image)
	print("Page ", pageNo, " - ", " C" if cValue >= 0.09 else "BW", " - ", cValue)
# 	os.remove(os.path.join(directory, image))
# os.rmdir(directory)