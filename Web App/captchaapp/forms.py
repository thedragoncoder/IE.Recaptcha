from django import forms
from .models import Inference
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np


class InferenceForm(forms.ModelForm):
    class Meta:
        model = Inference
        fields = ('image', 'prediction')

    def create(self, validated_data):
        inference_obj = Inference.objects.create(**validated_data)

        # HERE WHERE YOU CAN PUT YOUR OWN MODEL PATH

        model_json_path = 'captchaapp/Recaptcha_CNN_1.h5'
        model_weight_path = 'captchaapp/model.json'

        # load the image file an turn it to a numpy array
        img = image.load_img(inference_obj.image.file.filename, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array.shape = (1, 150, 150, 3)

        # load model model_architecture
        self.update_state(state='load model', meta={'progress': 25})
        with open('captchaapp/model.json', 'r') as f:
            model = model_from_json(f.read())
        # load model weights
        model.load_weights('captchaapp/Recaptcha_CNN_1.h5')

        # make image prediction
        prediction = model.predict(img_array, verbose=1)
        print(prediction)
        if prediction == 0:
            result = 'Normal'
        elif prediction == 1:
            result = 'PNEUMONIA'

        inference_obj.prediction = result
        return inference_obj
