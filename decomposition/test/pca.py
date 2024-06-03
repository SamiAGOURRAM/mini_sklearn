import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as skPCA
from decomposition.pca import PCA

X, _ = load_iris(return_X_y=True)
pca1 = PCA().fit(X)
Xt1 = pca1.transform(X)
Xinv1 = pca1.inverse_transform(Xt1)
pca2 = skPCA(n_components=2).fit(X)
Xt2 = pca2.transform(X)
Xinv2 = pca2.inverse_transform(Xt2)


for i in range(pca1.components_.shape[0]):
    assert np.allclose(pca1.components_[i], pca2.components_[i]) or np.allclose(pca1.components_[i], -pca2.components_[i])
assert np.allclose(pca1.explained_variance_ratio_, pca2.explained_variance_ratio_)
for i in range(Xt1.shape[1]):
    assert np.allclose(Xt1[:, i], Xt2[:, i]) or np.allclose(Xt1[:, i], -Xt2[:, i])
assert np.allclose(Xinv1, Xinv2)

print("Test passed!")