from utils import extract
import numpy as np
import tensorflow as tf
import keras.utils as image
import os

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
	# help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

test_dir = os.path.join('models','test')
features=['jaw','left_eye','left_undereye','mouth','nose','right_eye','right_undereye']
IMG_SIZE=(224,224)


model_eye_path = os.path.join('models','eye','eye.h5')
model_undereye_path = os.path.join('models','undereye','undereye.h5')
model_jaw_path = os.path.join('models','jaw','jaw.h5')
model_nose_path = os.path.join('models','nose','nose.h5')
model_mouth_path = os.path.join('models','mouth','mouth.h5')

os.makedirs(test_dir,exist_ok=True)
# IMG_PATH='images/train/not_tired/2.JPG'
IMG_PATH='images/train/tired/6.JPG'


# try:
    
eye=tf.keras.models.load_model(model_eye_path)
undereye=tf.keras.models.load_model(model_undereye_path)
jaw=tf.keras.models.load_model(model_jaw_path)
nose=tf.keras.models.load_model(model_nose_path)
mouth=tf.keras.models.load_model(model_mouth_path)

models={
	'jaw':(jaw,'jaw'),
	'mouth':(mouth,'mouth'),
	'left_eye':(eye,'eye'),
	'right_eye':(eye,'eye'),
	'left_undereye':(undereye,'undereye'),
	'right_undereye':(undereye,'undereye'),
	'nose':(nose,'nose')
}

pred={
	'jaw':[],
	'mouth':[],
	'eye':[],
	'undereye':[],
	'nose':[]
}

# filename=extract(save_dir=test_dir,target_image=IMG_PATH)
# file=os.path.join(test_dir,'left_eye',f'{filename}')
# img=image.load_img(file,target_size=IMG_SIZE)	
# img=image.img_to_array(img)
# img=img/255.0
# img=np.expand_dims(img,axis=0)
# # pred=models['left_undereye'][0].predict(img)
# # print(models['left_eye'][1])
# p=models['left_eye'][0].predict(img)
# pred[models['left_eye'][1]].append(p)
# # print(file)
# print(pred)


try:
	filename=extract(save_dir=test_dir,target_image=IMG_PATH)

	for feature in features:
		# model=models[feature][0]
		file=os.path.join(test_dir,f'{feature}',f'{filename}')
		img=image.load_img(file,target_size=IMG_SIZE)	
		img=image.img_to_array(img)
		img=img/255.0
		img=np.expand_dims(img,axis=0)
		pred[models[feature][1]].append(models[feature][0].predict(img))
	# print(file)
	# print(pred)
except Exception as e:
	print(e)


predictions=[]
for value in pred.values():
	if len(value)>1:
		# print(type(np.average(value)))
		predictions.append([np.average(value)])
	else:
		predictions.append(value[0])
print(predictions)
print(np.average(predictions))









