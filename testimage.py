from keras.models import load_model
import cv2
import numpy as np

model = load_model('model-048-0.027035.h5')

image = cv2.imread('t2.jpg')
image = np.asarray(image)
image = cv2.resize(image, (100, 100))
image = np.array([image])

print(model.summary())

x = model.predict(image, batch_size=1)

t=list(x[0])
print(t)
if(t[0]>t[1]):
    print("It is a signature")
else:
    print("It is not a signature")
