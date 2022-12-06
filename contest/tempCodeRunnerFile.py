
def x2X(x, R):
    B, x_length, x_width = x.shape
    dx = x_length - R + 1
    dy = x_width - R + 1
    altx = np.zeros((B, R, R, dx, dy))
    for i in range(R):
        for j in range(R):
            altx[:, i, j, :, :] = x[:, i:i+dx, j:j+dy]
    return altx.transpose(1, 2, 0, 3, 4).reshape(R*R, dx*dy*B)