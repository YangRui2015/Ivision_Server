import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import sqlite3
import io
import pickle
import torch

import numpy as np
from PIL import Image
from torchvision import models, transforms

from model import NIMA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "nima.pth"
base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

db_path = "user.db"
score_dict_path = "score_dict.pkl"

val_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


def file_to_numpy(f):
    in_memory_file = io.BytesIO()
    f.save(in_memory_file)
    return np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_score(img):
    new_img = val_transform(img).unsqueeze(0).to(device)
    outputs = model(new_img)
    dist = torch.arange(1, 11).float().to(device)
    p_mean = (outputs.view(-1, 10) * dist).sum(dim=1)
    return myremap(float(p_mean))


# def get_dist(img):
#     new_img = val_transform(img).unsqueeze(0).to(device)
#     outputs = model(new_img)
#     print(outputs.data, outputs.data.cpu())
#     # dist = torch.arange(1, 11).float().to(device)
#     # p_mean = (outputs.view(-1, 10) * dist).sum(dim=1)
#     np_dist = outputs.data.cpu().numpy()
#     print(np_dist)
#     result = {'dist': np_dist.tolist(), 'var': np.var(np_dist).tolist()}
#     return result


def remap(x):
    return sigmoid(x - 4.5) * 10


def myremap(x):    # 对分数做一个映射
    newscore = round(min(max((x - 3) * 100 / 3, 0), 100))
    return newscore


def rank(name, score, db_path="user.db"):
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE user (id CHAR(40) PRIMARY KEY, times INT,score INT);''')
        conn.commit()
        conn.close()
        print("finish make datebase")

    present_best_score = 0
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    result = cur.execute("select * from user where id = ?", (name,)).fetchone()
    if result and len(result) > 1:
        present_best_score = result[2]
        if score > present_best_score:
            cur.execute("update user set score = ? where id = ?",(score, name))
            present_best_score = score

    else:
        present_best_score = score
        cur.execute("insert into user(id, score) values (?,?)",(name, score))

    conn.commit()
    conn.close()

    # rank
    if os.path.exists(score_dict_path):
        with open(score_dict_path, "rb") as f:
            dict_mat = pickle.load(f)
        score_dict = dict_mat["score"]
        total = dict_mat["total"] + 1
    else:
        score_dict = {}
        total = 1

    if str(score) not in score_dict:
        score_dict[score] = 1
    else:
        score_dict[score] += 1

    precessor = 0
    for i in range(score + 1, 101):
        if i in score_dict:
            precessor += score_dict[i]
    my_rank = round((1 - precessor / total) * 100, 2)

    with open(score_dict_path, "wb") as f:
        pickle.dump({"score":score_dict, "total":total}, f)

    return present_best_score, my_rank


if __name__ == "__main__":
    path = "4.jpg"
    img = Image.open(path).convert("RGB")
    score = get_score(img)
    print(score)
    print(rank("杨瑞3", score))
