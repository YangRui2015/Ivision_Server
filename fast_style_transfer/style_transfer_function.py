# coding: utf-8
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from .preprocessing import preprocessing_factory
from . import reader
from . import model
#from preprocessing import preprocessing_factory
#import reader
#import model
import time
import os
import cv2
import numpy as np
from PIL import Image

basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)
modelpaths = [basedir + "/module/wave.ckpt-done",
             basedir + "/module/cubist.ckpt-done",
             basedir + "/module/mosaic.ckpt-done",
             basedir + "/module/feathers.ckpt-done",
             basedir + "/module/scream.ckpt-done",
             basedir + "/module/starry.ckpt-done",
             basedir + "/module/udnie.ckpt-done",
             ]


def style_transfer(imagepath, modelnum, savepath='generated/res.jpg'):
    modelpath = modelpaths[modelnum]
    # Get image's height and width.
    height = 0
    width = 0
    with open(imagepath, 'rb') as img:
        with tf.Session().as_default() as sess:
            if imagepath.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))

            if len(image.shape) == 3 and image.shape[2] > 3:         # 防止四维图像
                image = image[:,:,0:3]

            height = image.shape[0]
            width = image.shape[1]
    print('get image size: %dx%d' % (width, height))
    ratio = 1
    max_len = max(width, height)
    if max_len > 1000:
        ratio = 1000 / max_len
    image = cv2.resize(image, (int(ratio * width), int(ratio * height)))

    height, width = image.shape[0:2]
    print('change image size: %dx%d' % (width, height))
    mask = np.random.randint(-10,10,(height, width, 3))          # 加上随机噪声，发现一大片纯白色的会导致结果出现大片纯色结果

    image = image.astype(np.int32)
    image += np.int32(mask)
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)
    Image.fromarray(image).save(imagepath)
    import pdb
    pdb.set_trace()

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                "vgg_16",
                is_training=False)
            image = reader.get_image(imagepath, height, width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer()])
            # Use absolute path
            absmodelpath = os.path.abspath(modelpath)
            saver.restore(sess, absmodelpath)

            # Make sure 'generated' directory exists.
            # generated_file = 'generated/res.jpg'
            generated_file = savepath
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # Generate and write image data to file.
            with open(generated_file, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()
                print('Elapsed time: %fs' % (end_time - start_time))

                print('Done. Please check %s.' % generated_file)


if __name__ == "__main__":
    style_transfer("/data/Yangrui/Ivision_Sever/static/style_transfer/2018-12-16-18-18-34_0.14.jpg",1,"2018-12-16-18-18-34_0.14_t.jpg")
