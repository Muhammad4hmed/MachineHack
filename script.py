import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from use import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)



def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path):
    Dmax = 608
    Dmin = 288
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor


def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")



# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction



def get_string_from_image(test_image_path):
	vehicle, LpImg,cor = get_plate(test_image_path)
	if (len(LpImg)): #check if there is at least one license image
    # Scales, calculates absolute values, and converts the result to 8-bit.
	    plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
	    
	    # convert to grayscale and blur the image
	    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
	    blur = cv2.GaussianBlur(gray,(7,7),0)
	    
	    # Applied inversed thresh_binary 
	    binary = cv2.threshold(blur, 180, 255,
	                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	    
	    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)


	cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# creat a copy version "test_roi" of plat_image to draw bounding box
	test_roi = plate_image.copy()

	# Initialize a list which will be used to append charater image
	crop_characters = []

	# define standard width and height of character
	digit_w, digit_h = 30, 60

	for c in sort_contours(cont):
	    (x, y, w, h) = cv2.boundingRect(c)
	    ratio = h/w
	    if 1<=ratio<=3.5: # Only select contour with defined ratio
	        if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
	            # Draw bounding box arroung digit number
	            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

	            # Sperate number and gibe prediction
	            curr_num = thre_mor[y:y+h,x:x+w]
	            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
	            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	            crop_characters.append(curr_num)

	'''print("Detect {} letters...".format(len(crop_characters)))
	fig = plt.figure(figsize=(10,6))
	plt.axis(False)
	plt.imshow(test_roi)'''
	#plt.savefig('grab_digit_contour.png',dpi=300)


	fig = plt.figure(figsize=(14,4))
	grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)


	



	fig = plt.figure(figsize=(15,3))
	cols = len(crop_characters)
	grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

	final_string = ''
	for i,character in enumerate(crop_characters):
	    fig.add_subplot(grid[i])
	    title = np.array2string(predict_from_model(character,model,labels))
	    plt.title('{}'.format(title.strip("'[]"),fontsize=20))
	    final_string+=title.strip("'[]")
	    #plt.axis(True)
	    plt.imshow(character,cmap='gray')

	return final_string


#get_string_from_image(test_image_path)



TEST_PATH='sample_videos/frames'

for count,files in enumerate(os.listdir(TEST_PATH)):
	path=os.path.join(TEST_PATH,files)
	print(get_string_from_image(path))
	if count==15:
		break


def mp4tojpg(vid_path): 
# Opens the Video file
	cap= cv2.VideoCapture(vid_path)
	i=0
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == False:
			break
		if i%60==0 and i!=0:
			cv2.imwrite('sample_videos/frames/{}.jpg'.format(i),frame)
		i+=1
 
	cap.release()
	cv2.destroyAllWindows()


vid_path='sample_videos/2B.mp4'
#mp4tojpg(vid_path)