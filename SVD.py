import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD


X, y = load_digits(return_X_y=True)

imag=X[0]
image=np.asarray(imag)
#print (image)
#reshaping the array
image = image.reshape((8, 8))

#print the original image from the dataset
plt.matshow(image, cmap = 'gray')
plt.show()

U, s, VT = np.linalg.svd(image)
Sigma = np.zeros((image.shape[0], image.shape[1]))
Sigma[:image.shape[0], :image.shape[0]] = np.diag(s)
n_components = 2
Sigma = Sigma[:, :n_components]
VT = VT[:n_components, :]

#A is the final matrix
A = U.dot(Sigma.dot(VT))
print(A)


svd = TruncatedSVD(n_components=2)
X_red = svd.fit_transform(X)
X_red[0]

image_compressed = svd.inverse_transform(X_red[0].reshape(1,-1))
image_compressed = image_compressed.reshape((8,8))

#prints the final compressed image
plt.matshow(image_compressed, cmap = 'gray')
plt.show()
