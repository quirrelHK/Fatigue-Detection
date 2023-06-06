from flask import Flask, render_template, request
from utils import extract
import numpy as np
import tensorflow as tf
import keras.utils as image
from keras.models import load_model
import os

app = Flask(__name__)

def load_models():
    model_eye_path = os.path.join('models', 'eye', 'eye.h5')
    model_undereye_path = os.path.join('models', 'undereye', 'undereye.h5')
    model_jaw_path = os.path.join('models', 'jaw', 'jaw.h5')
    model_nose_path = os.path.join('models', 'nose', 'nose.h5')
    model_mouth_path = os.path.join('models', 'mouth', 'mouth.h5')
    
    eye = load_model(model_eye_path)
    undereye = load_model(model_undereye_path)
    jaw = load_model(model_jaw_path)
    nose = load_model(model_nose_path)
    mouth = load_model(model_mouth_path)
    
    models = {
        'jaw': (jaw, 'jaw'),
        'mouth': (mouth, 'mouth'),
        'left_eye': (eye, 'eye'),
        'right_eye': (eye, 'eye'),
        'left_undereye': (undereye, 'undereye'),
        'right_undereye': (undereye, 'undereye'),
        'nose': (nose, 'nose')
    }
    
    return models


def predict_class(model,file):
    IMG_SIZE = (224, 224)
    img = image.load_img(file, target_size=IMG_SIZE)    
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    avg_prediction=None
    models = load_models()
    test_dir = os.path.join('models', 'test')
    features = ['jaw', 'left_eye', 'left_undereye', 'mouth', 'nose', 'right_eye', 'right_undereye']
    
    
    img_file = request.files['image']
    os.makedirs(test_dir,exist_ok=True)
    img_path = os.path.join(test_dir, 'temp.jpg')
    img_file.save(img_path)
    
    pred = {'jaw':[],
            'mouth':[],
            'undereye':[],
            'eye':[],
            'nose':[]}
 
    try:
        # print(test_dir,img_path)
        filename = extract(save_dir=test_dir, target_image=img_path)
    
        for feature in features:
            file = os.path.join(test_dir, feature, filename)
            prediction = predict_class(models[feature][0],file)
            pred[models[feature][1]].append(prediction)           
        # print(pred)
               
    
    
        predictions = {}
        pred_cnt = [0,0]
        for key,value in pred.items():
            if len(value) > 1:
                # predictions.append([np.average(value)])
                # pred[key] = np.average(value)
                predictions[key] = [np.average(value)]
                
            else:
               
                predictions[key] = value
            if predictions[key][0] > 0.5:
                pred_cnt[1]+=1
            else:
                pred_cnt[0]+=1
        
        print(predictions)
        avg_prediction = [x for x in predictions.values()]
        
        avg_prediction = np.average(avg_prediction)
        cnt = 1 if pred_cnt[1] > pred_cnt[0] else 0
        
    except Exception as e:
        print(e)
    
    
    return render_template('result.html', prediction=avg_prediction, pred_cnt=cnt)


if __name__ =='__main__':
    app.run(debug=True)
