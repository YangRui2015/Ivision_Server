from fast_style_transfer.style_transfer_function import style_transfer


def test(path, number, path1):
    style_transfer(path, number, path1)


if __name__=="__main__":
    test("/data/Yangrui/Ivision_Sever/static/style_transfer/2018-12-16-20-28-27_0.942.jpg",1,"/data/Yangrui/Ivision_Sever/static/style_transfer/2018-12-16-20-28-27_0.942_t.jpg")