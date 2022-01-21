from keras.models import load_model
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2 as cv

#im = imageio.imread("a3Rql9C.png")
#im = cv.imread("testDG.png")
im = cv.imread("three.png")
#gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.show()

# reshape and normalize the image
gray[:4, :] = 0
gray[:, :4] = 0
gray[-4:, :] = 0
gray[:, -4:] = 0
cv.imshow('gray', gray)
cv.waitKey(0)
num_pixels = gray.shape[0] * gray.shape[1]
gray1 = gray.reshape((1, 28 * 28)).astype('float32')
gray2 = gray1 / 255

# load the model
model = load_model("test_model.h5")

# predict digit
prediction = model.predict(gray2)
print(prediction.argmax())
