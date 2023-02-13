import numpy as np

from sklearn.preprocessing import StandardScaler

from bob.fusion.base.preprocessor import Tanh


def test_tanh_preprocessor():
    DATA = [[0, 0], [0, 0], [1, 1], [1, 1]]
    Y = np.ones(len(DATA), dtype=bool)
    MEAN = [0.5, 0.5]
    DATA_T = [[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]
    DATA2 = [[2, 2]]
    DATA2_T = [[3.0, 3.0]]
    scaler = StandardScaler()
    scaler.fit(np.asarray(DATA))
    assert np.allclose(scaler.mean_, MEAN)
    assert np.allclose(scaler.transform(DATA), DATA_T)
    assert np.allclose(scaler.transform(DATA2), DATA2_T)

    def tanh(s, mean, std):
        return 0.5 * (np.tanh(0.01 * (s - mean) / std) + 1)

    scaler = Tanh()
    scaler.fit(DATA, y=Y)
    assert np.allclose(scaler.mean_, MEAN)
    assert np.allclose(
        scaler.transform(DATA), tanh(DATA, scaler.mean_, scaler.scale_)
    )
    assert np.allclose(
        scaler.transform(DATA2), tanh(DATA2, scaler.mean_, scaler.scale_)
    )

    assert np.allclose(scaler.inverse_transform(scaler.transform(DATA2)), DATA2)

    assert np.allclose(scaler.inverse_transform(scaler.transform(DATA)), DATA)

    scaler = Tanh()
    scaler.fit_transform(DATA, y=Y)
    assert np.allclose(scaler.mean_, MEAN)
    assert np.allclose(
        scaler.transform(DATA), tanh(DATA, scaler.mean_, scaler.scale_)
    )
    assert np.allclose(
        scaler.transform(DATA2), tanh(DATA2, scaler.mean_, scaler.scale_)
    )

    assert np.allclose(scaler.inverse_transform(scaler.transform(DATA2)), DATA2)

    assert np.allclose(scaler.inverse_transform(scaler.transform(DATA)), DATA)
