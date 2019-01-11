import requests

url = "https://ivision.mynatapp.cc" + "/"   # 添加路径名

imgpath = ""
files={'img':('test.png',open(imgpath,'rb'),'image/png')}
res = requests.request("POST", url, files)
print(res)