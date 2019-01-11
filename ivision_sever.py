# -*- coding: utf-8 -*-
from flask import Flask
from flask import request
import requests
import os
import sqlite3
import numpy as np
import random
import cv2
from PIL import Image
import pickle
import time
import requests
from deep_dream.app_run import run_app


app = Flask(__name__)
print(app.static_folder)

basedir = os.path.abspath(os.path.dirname(__file__))
images_path = basedir + '/static/'
webpath = "https://ivision.mynatapp.cc"


wexinpath = "https://api.weixin.qq.com/sns/jscode2session"
appid = "wx7e7c46191efb1db8"
secret = "fc3c1710de3b2b1cd705d2833904d8e8"

# picklefile = open("dict.pkl", "rb")
# face_dict = pickle.load(picklefile)
# pic_dict = pickle.load(picklefile)
# picklefile.close()


@app.route('/getinfo', methods=['POST','GET'])
def getinfo():
    code = request.form['code']
    url = wexinpath + "?appid=" + appid + "&secret=" + secret + "&js_code=" + code + "&grant_type=authorization_code"
    ret = requests.get(url)
    openid = ret.json()["openid"]

    conn = sqlite3.connect("user.db")
    cur = conn.cursor()
    result = cur.execute("select * from user where id = ?",(openid,)).fetchone()
    # print(result)
    if result:
        times = result[1]
        if not times:
            times = 0
        times += 1
        cur.execute("update user set times = ? where id = ?", (times, openid))    # times是登录次数
    else:
        cur.execute("insert into user(id, times) values (?,?)",(openid, 1))

    conn.commit()
    conn.close()
    return openid


@app.route('/Assessment', methods=['POST','GET'])
def assessment():
    print("Assessment begin")
    openid = request.headers["user"]
    print(openid)

    img = request.files.get('file')
    path = images_path + "assessment/" + time.strftime('%Y-%m-%d-%H-%M-%S_', time.localtime(time.time())) + str(round(random.random(), 3)) + ".jpg"
    img.save(path)
    
    # 发送请求
    url = "http://127.0.0.1:8081" + "/score"   # 添加路径名
    headers = {"user":openid,"path":path}
    res = requests.post(url,headers=headers)
    print(res.text)
    string = res.text
    #:string = str(score) + " " + str(present_best_score) + " " + str(your_rank)
    return string


@app.route('/style', methods=['POST'])
def transtyle():
    print('style session: style transfer')
    openid = request.headers["user"]
    print(openid)
    num = int(request.headers["Model"])
    if num is None:
        num = 0

    img = request.files.get('file')
    path = images_path + "style_transfer/" + time.strftime('%Y-%m-%d-%H-%M-%S_', time.localtime(time.time())) + str(round(random.random(), 3)) + ".jpg"
    img.save(path)
    print(img)

    # 风格图片生成
    url = "http://127.0.0.1:8080" + "/style"   # 添加路径名
    files = {'file': open(path, 'rb')}
    headers = {"user":openid, "Model":str(num), "path":path}
    res = requests.post(url, files=files, headers=headers)
    new_path = res.text
    new_path = new_path.replace("https://2333.mynatapp.cc", "/home/yangrui/WeiSever")
    print(new_path)
    new_path_2 = path.replace("/home/yangrui/WeiSever", "/data/Yangrui/Ivision_Sever")
    new_path_2 = new_path_2.replace(".jpg", "_t.jpg")
    img = Image.open(new_path)
    img.save(new_path_2)

    # 略缩图生成
    max_l = max(img.size[0], img.size[1])
    if max_l > 500:
        ratio = 500 / max_l
        img = img.resize((int(ratio * img.size[0]), int(ratio * img.size[1])))
    small_path = new_path_2.replace("_t.jpg", "_s.jpg")
    img.save(small_path)

    return new_path_2.replace(images_path, "")


@app.route('/cartoon', methods=['POST'])
def cartoon():
    print('cartoon session: ')
    openid = request.headers["user"]
    print(openid)
    num = int(request.headers["Model"])
    if num is None:
        num = 0

    img = request.files.get('file')
    path = images_path + "cartoon/" + time.strftime('%Y-%m-%d-%H-%M-%S_', time.localtime(time.time())) + str(round(random.random(), 3)) + ".jpg"
    img.save(path)
    print(img)

    # 风格图片生成
    url = "http://127.0.0.1:8082" + "/cartoon"   # 添加路径名
    headers = {"path":path,"Model":str(num)}
    res = requests.post(url,headers=headers)
    new_path = res.text
    print(new_path)
    img = Image.open(new_path)

    # 略缩图生成
    max_l = max(img.size[0], img.size[1])
    if max_l > 500:
        ratio = 500 / max_l
        img = img.resize((int(ratio * img.size[0]), int(ratio * img.size[1])))
    small_path = new_path.replace("_t.jpg","_s.jpg")
    img.save(small_path)

    return new_path.replace(images_path,"")


@app.route('/dream', methods=['POST'])
def deep_dream():
    print('dream session: dream transfer')
    openid = request.headers["user"]
    print(openid)

    img = request.files.get('file')
    path = images_path + "deep_dream/" + time.strftime('%Y-%m-%d-%H-%M-%S_', time.localtime(time.time())) + str(round(random.random(), 3)) + ".jpg"
    img.save(path)
    print(img)

    # deep dream 图片生成
    new_path = path.replace(".jpg","_t.jpg")
    print(path,new_path)
    run_app(path,  new_path)

    # 略缩图生成
    img_new = Image.open(new_path)
    max_l = max(img_new.size[0], img_new.size[1])
    if max_l > 500:
        ratio = 500 / max_l
        img_new = img_new.resize((int(ratio * img_new.size[0]), int(ratio * img_new.size[1])))
    small_path = new_path.replace("_t.jpg","_s.jpg")
    img_new.save(small_path)

    return new_path.replace(images_path,"")


@app.route('/send', methods=['POST'])
def getmessage():
    print('message session: get message')
    message = request.form["message"]

    with open("suggestions.txt", "a") as f:
        f.write(message + "\n")

    return "我们已经收到了你的建议！"


if __name__ == '__main__':
    app.run('127.0.0.1', port=8083, debug=True)
