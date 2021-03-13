from django.shortcuts import render
from keras.applications.imagenet_utils import decode_predictions
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model_json_path = 'captchaapp/Recaptcha_CNN_1.h5'
model_weight_path = 'captchaapp/model.json'


def index(request):
    if request.method == "POST":
        with open('captchaapp/model.json', 'r') as f:
            model = model_from_json(f.read())
        # load model weights
        model.load_weights('captchaapp/Recaptcha_CNN_1.h5')

        f = request.FILES['sentFile']  # here you get the files needed
        path = "captchas/"
        actual = f.name[:4]
        print(actual)
        path = path + f.name
        print(path)
        response = {}  # print(f)
        # dis = Image.read(f)
        im = plt.imread(f)
        original = []
        original.append(im)
        arr = np.asarray(original)
        prediction = model.predict(arr)
        y_pred = tf.math.argmax(prediction, axis=-1)
        # label = decode_predictions(prediction)
        answer = ('{}'.format(
            ''.join(map(str, y_pred[0].numpy()))))
        response['img'] = "../media/" + f.name
        response['answer'] = answer
        response['actual'] = actual
        return render(request, 'index.html', response)
    else:
        return render(request, 'index.html')
