
import sys
import tensorflow as tf
from PIL import Image, ImageFilter

def give_output(imvalue):
    
    a = tf.placeholder(tf.float32, [None, 784])
    c = tf.Variable(tf.zeros([784, 10]))
    f = tf.Variable(tf.zeros([10]))
    
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
       
    def conv2d(a, c):
      return tf.nn.conv2d(a, c, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(a):
      return tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   
    
    c_conv1 = weight_variable([5, 5, 1, 32])
    f_conv1 = bias_variable([32])
    
    imgA = tf.reshape(a, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(imgA, c_conv1) + f_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    f_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, c_conv2) + f_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    c_fc1 = weight_variable([7 * 7 * 64, 1024])
    f_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, c_fc1) + f_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    c_fc2 = weight_variable([1024, 10])
    f_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, c_fc2) + f_fc2)
    
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
   
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model2.ckpt")
        #print ("Model restored.")
       
        prediction=tf.argmax(y_conv,1)
        return prediction.eval(feed_dict={a: [imvalue],keep_prob: 1.0}, session=sess)


def imageprepare(argv):
    
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) 
    
    if width > height: 
        
        nheight = int(round((20.0/width*height),0)) 
        if (nheigth == 0):
            nheigth = 1  
       
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop))
    else:
         
        nwidth = int(round((20.0/height*width),0))
        if (nwidth == 0): 
            nwidth = 1
         
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) 
        newImage.paste(img, (wleft, 4))
    
   

    tv = list(newImage.getdata()) 
    
  
    tva = [ (255-a)*1.0/255.0 for a in tv] 
    return tva
   

def main(argv):
    
    imvalue = imageprepare(argv)
    predint = predictint(imvalue)
    print (predint[0])
    
if __name__ == "__main__":
    main(sys.argv[1])
