import numpy as np

LR = 0.1
w = np.array([0, 0, 0, 0, 0, 0])


def loss_hinge(x, y, w):
    return np.max(1 - np.dot(w, x) * y, 0)


def delta_loss_hinge(x, y, w):
    if np.dot(w, x) * y < 1:
        delta = np.array([0, 0, 0, 0, 0, 0])
        for i in range(0, 6):
            delta[i] = -x[i] * y
        return delta
    return 0


x = np.array([[1, 0, 1, 0, 0, 0],
             [0, 1, 0, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 1]])
y = np.array([-1, 1, -1, 1])

for epoch in range(0, 4):
    w = w - LR*delta_loss_hinge(x[epoch], y[epoch], w)
    print(w)

print(w)
# print(np.dot(x,w))
