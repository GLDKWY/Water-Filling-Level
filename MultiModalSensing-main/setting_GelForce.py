
def init():
    global RESCALE, N_, M_, x0_, y0_, dx_, dy_, fps_
    RESCALE = 1

    """
    N_, M_: the row and column of the marker array
    x0_, y0_: the coordinate of upper-left marker (in original size)
    dx_, dy_: the horizontal and vertical interval between adjacent markers (in original size)
    fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds
    """
    N_ = 10
    M_ = 4
    fps_ = 24
    x0_ = 152 / RESCALE  # 40
    y0_ = 172 / RESCALE  # 22
    dx_ = 80 / RESCALE
    dy_ = 95 / RESCALE
