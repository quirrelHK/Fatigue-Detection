from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os


skip = {'inner_mouth','right_eyebrow','left_eyebrow'}
counter = {
    'right_eye':0,
    'left_eye':0,
    'left_undereye':0,
    'right_undereye':0,
    'mouth':0,
    'nose':0,
    'jaw':0
}
SHAPE_PREDICTOR='shape_predictor_68_face_landmarks.dat'


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
def extract(save_dir, target_image):
    for k in counter.keys():
        os.makedirs(os.path.join(save_dir,k),exist_ok=True)
    # for num,filename in enumerate(os.listdir(args['image']), start=1):
    try:
        filename=target_image.split('\\')[-1]
        filename=filename.split('.')[0]
        # print(filename)
        image=cv2.imread(target_image)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray, 1)
        
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            nose_box = []
            for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                if name in skip: continue
        
                (x,y,w,h) = cv2.boundingRect(np.array([shape[i:j]]))
                if name == 'jaw':
                    if not nose_box: continue

                    h+=y
                    y=nose_box[0][-3]+nose_box[0][-1]
                    h-=y
                roi = image[y:y+h, x:x +w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        
            
                # counter[name] = counter.get(name)+1
                # cv2.imwrite(os.path.join(save_dir,f'{name}',f'{counter}.jpg'), roi)
                path=os.path.join(save_dir,f'{name}')
                os.makedirs(path,exist_ok=True)
                path = os.path.join(path,f'{filename}.jpg')
                cv2.imwrite(path, roi)
                
                if name == 'nose':
                    nose_box.append((x,y,w,h))
        
            
                if name=='left_eye':
                
                    name='left_undereye'
                    yn=y+h
                    roi2 = image[yn:yn+h+h, x:x+w]
                    roi2 = imutils.resize(roi2, width=250, inter=cv2.INTER_CUBIC)
        
                    # counter[name] = counter.get(name)+1
                    # cv2.imwrite(os.path.join(save_dir,f'{name}',f'{counter[name]}.jpg'), roi2)
                    # cv2.imwrite(os.path.join(save_dir,f'{name}',f'{filename}.jpg'), roi)
                    path=os.path.join(save_dir,f'{name}')
                    os.makedirs(path,exist_ok=True)
                    path = os.path.join(path,f'{filename}.jpg')
                    cv2.imwrite(path, roi2)

                
                if name=='right_eye':
                
                    name='right_undereye'
                    yn=y+h
                    roi2 = image[yn:yn+h+h, x:x+w]
                    roi2 = imutils.resize(roi2, width=250, inter=cv2.INTER_CUBIC)


                    # counter[name] = counter.get(name)+1
                    # cv2.imwrite(os.path.join(save_dir,f'{name}',f'{counter[name]}.jpg'), roi2)
                    # cv2.imwrite(os.path.join(save_dir,f'{name}',f'{filename}.jpg'), roi)
                    path=os.path.join(save_dir,f'{name}')
                    os.makedirs(path,exist_ok=True)
                    path = os.path.join(path,f'{filename}.jpg')
                    cv2.imwrite(path, roi2)
        return filename+'.jpg'

    except Exception as e:
            # print(filename)
        print(e)

    

if __name__ == '__main__':
    img_path='images/train/not_tired/3.JPG'
    test=os.path.join('models','test')
    extract(test,img_path)