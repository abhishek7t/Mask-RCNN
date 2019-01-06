""" Mask RCNN
Pedestrian and Car Dataset
CIS 680: "Vision and Learning"
https://fling.seas.upenn.edu/~cis680/wiki/index.php?title=CIS_680:_Vision_%26_Learning
Abhishek Tiwary (amannitr@cis.upenn.edu)
Homework 02b
INSTRUCTIONS : 
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import time 

import tensorflow as tf
import numpy as np
import pdb

# import utils - Segmentation fault
import data_utils
import layers
import utils
import spatial_transformer


class RPN(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 4
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 2
        self.skip_step = 200
        self.n_test = 400
        self.training= tf.Variable(False)
        self.status = False

    def get_data(self):
        with tf.name_scope('data'):
            self.anchor = np.array([-1,-1,2,2])
            train_data, test_data = data_utils.get_dataset(self.batch_size, anchor = self.anchor)
           
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                        train_data.output_shapes)
            img, self.people_label, self.car_label, self.iou_scores, self.bbox_matrix, self.tx_star, self.ty_star, self.tw_star, self.th_star, self.label, self.people_mask, self.car_mask = iterator.get_next()
         
            self.img = img
            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)
           

    def proposal(self):
        '''
        Use ground truth bounding boxes to find iou for different anchor locations
        1: object, 0: not object, -1: ignore
        '''
        self.proposal_scores = self.iou_scores


    def inference(self):
        self.base_network = layers.base_network(self.img, self.training, 'base_network')
        self.intermediate_layer = layers.intermediate_layer(self.base_network, self.training, 'intermediate_layer')
        self.logits_cls = layers.clf_layer(self.intermediate_layer, self.training,'cls_layer')
        self.scores_cls = tf.nn.sigmoid(self.logits_cls)

        reg = layers.reg_layer(self.intermediate_layer,self.training, 'reg_layer')
        
        self.tx, self.ty, self.tw, self.th = self.parameterize(reg)

        # Faster RCNN additional layers
        scores_cls_flat = tf.reshape(self.scores_cls,[-1, self.scores_cls.shape[1]*self.scores_cls.shape[2]])
        
        # Find the top 2 iou-score locations in each of the batch
        self.values, self.indices = tf.nn.top_k(scores_cls_flat, k=2, sorted=True, name=None)
        self.ind1, self.ind2 = self.indices[:,0:1], self.indices[:,1:2]
        self.ind1 = tf.concat([tf.reshape(tf.range(self.batch_size),[-1,1]), self.ind1], 1)
        self.ind2 = tf.concat([tf.reshape(tf.range(self.batch_size),[-1,1]), self.ind2], 1)
        
        x1, y1, w1, h1 = self.gather(reg[:,:,:,0:1], self.ind1), self.gather(reg[:,:,:,1:2], self.ind1), self.gather(reg[:,:,:,2:3], self.ind1), self.gather(reg[:,:,:,3:4], self.ind1)
        x2, y2, w2, h2 = self.gather(reg[:,:,:,0:1], self.ind2), self.gather(reg[:,:,:,1:2], self.ind2), self.gather(reg[:,:,:,2:3], self.ind2), self.gather(reg[:,:,:,3:4], self.ind2)
        
        x, y, w, h = tf.concat([x1,x2], axis=0), tf.concat([y1,y2], axis=0), tf.concat([w1,w2], axis=0), tf.concat([h1,h2], axis=0)
        # x,y,w,h = x1, y1, w1, h1
        theta = tf.concat([w*16/128.0, 0.0*w, (x*16 - 64)/64.0, 0.0*h, h*16/128, (y*16 - 64)/64.0],axis=1)
        img = tf.concat([self.base_network, self.base_network], 0)
        # img = self.img
        label1, label2 = self.gather(self.label, self.ind1), self.gather(self.label, self.ind2)
        label = tf.concat([label1, label2], 0)
        # label = label1
        label = tf.one_hot(label, self.n_classes, on_value=1.0, off_value=0.0, axis=-1)
        self.one_hot_label = tf.reshape(label, [-1, self.n_classes])
       
        spatial_transformer_out = spatial_transformer.transformer(img, theta, out_size=(4,4))
        spatial_transformer_out = tf.reshape(spatial_transformer_out, [-1,4,4,128])
       
        self.logits = layers.faster_rcnn(spatial_transformer_out, self.training, 'faster_rcnn',self.n_classes)

        # Mask RCNN
        # self.logits_mask = layers.mask_rcnn(spatial_transformer_out, self.training, 'mask_rcnn',self.n_classes)
        # Choose the correct ground truth mask for each example
        mask1 = tf.where(label1 == 0, self.people_mask, self.car_mask)
        mask2 = tf.where(label2 == 0, self.people_mask, self.car_mask)
        self.mask = tf.concat([mask1, mask2], 0)
        
        mask_train = tf.concat([self.people_mask, self.car_mask], 0)
        x_train = tf.concat([self.people_label[:,0:1] + self.people_label[:,2:3]/2.0 , self.car_label[:,0:1] + self.car_label[:,2:3]/2.0], 0)
        y_train = tf.concat([self.people_label[:, 1:2] + self.people_label[:, 3:4]/2.0 , self.car_label[:, 1:2] + self.car_label[:,3:4]/2.0], 0)
        w_train = tf.concat([self.people_label[:,2:3] , self.car_label[:,2:3]], 0)
        h_train = tf.concat([self.people_label[:,3:4] , self.car_label[:,3:4]], 0)
        theta_train = tf.concat([w_train*16/128.0, 0.0*w_train, (x_train*16 - 64)/64.0, 0.0*h_train, h_train*16/128, (y_train*16 - 64)/64.0],axis=1)
        
        spatial_transformer_out_train = spatial_transformer.transformer(img, theta_train, out_size=(4,4))
        spatial_transformer_out_train = tf.reshape(spatial_transformer_out_train, [-1,4,4,128])
        # mask_logits_train = layers.mask_rcnn(spatial_transformer_out_train, self.training, 'mask_rcnn',self.n_classes)

        self.mask = tf.cond(self.training, lambda: mask_train, lambda: self.mask)
        # self.logits_mask = tf.cond(self.training, lambda: mask_logits_train, lambda: self.logits_mask)
        spatial_transformer_out = tf.cond(self.training, lambda: spatial_transformer_out_train, lambda: spatial_transformer_out)
        self.logits_mask = layers.mask_rcnn(spatial_transformer_out, self.training, 'mask_rcnn',self.n_classes)

        # logits_train = layers.faster_rcnn(spatial_transformer_out_train, self.training, 'faster_rcnn',self.n_classes)
        # self.logits = tf.cond(self.training, lambda: logits_train, lambda: self.logits)
        # self.logits = layers.faster_rcnn(spatial_transformer_out, self.training, 'faster_rcnn',self.n_classes)
        # label_train = tf.concat([label1 * 0, label2*0 + 1], 0)
        # label_train = tf.one_hot(label_train, self.n_classes, on_value=1.0, off_value=0.0, axis=-1)
        # one_hot_label_train = tf.reshape(label_train, [-1, self.n_classes])
        # self.one_hot_label = tf.cond(self.training, lambda: one_hot_label_train, lambda: self.one_hot_label)
        print('hi!')

    def gather(self,t,ind):
        # Flatten the tensor t
        t_flat = tf.reshape(t,[-1, t.shape[1]*t.shape[2]])
        t1 = tf.gather_nd(t_flat, ind, name=None)
        
        return tf.expand_dims(t1,-1)

    def parameterize(self, reg):
        w_a, h_a = self.anchor[2] + 1, self.anchor[3] + 1
        tx = (reg[:,:,:,0] - self.bbox_matrix[:,:,:,0])/w_a
        ty = (reg[:,:,:,1] - self.bbox_matrix[:,:,:,1])/w_a
        tw = tf.log(1e-9 + reg[:,:,:,2] / w_a)
        th = tf.log(1e-9 + reg[:,:,:,3] / h_a)

        tx = tf.reshape(tx, [-1, tx.shape[1], tx.shape[2], 1])
        ty = tf.reshape(ty, [-1, ty.shape[1], ty.shape[2], 1])
        tw = tf.reshape(tw, [-1, tw.shape[1], tw.shape[2], 1])
        th = tf.reshape(th, [-1, th.shape[1], th.shape[2], 1])

        tx = tf.where(tf.is_nan(tx), tf.zeros_like(tx), tx)
        ty = tf.where(tf.is_nan(ty), tf.zeros_like(ty), ty)
        tw = tf.where(tf.is_nan(tw), tf.zeros_like(tw), tw)
        th = tf.where(tf.is_nan(th), tf.zeros_like(th), th)

        tx = tf.where(tf.is_inf(tx), tf.zeros_like(tx), tx)
        ty = tf.where(tf.is_inf(ty), tf.zeros_like(ty), ty)
        tw = tf.where(tf.is_inf(tw), tf.zeros_like(tw), tw)
        th = tf.where(tf.is_inf(th), tf.zeros_like(th), th)

        return tx, ty, tw, th
        

    def loss(self):
        '''
        define loss function
        use sigmoid_cross_entropy loss for cls
        use smooth_L1 for reg
        Mask RCNN multi task loss, L = L_cls + L_box + L_mask
        '''
        with tf.name_scope('loss'):
     
            valid_pixels = tf.where(self.proposal_scores < -.1, tf.zeros_like(self.proposal_scores), tf.ones_like(self.proposal_scores))
            entropy_cls = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.proposal_scores, tf.float32), logits=self.logits_cls)
        
            entropy_cls = valid_pixels * entropy_cls
            factor = 64 * self.batch_size / tf.reduce_sum(valid_pixels) 
            self.loss_cls = tf.reduce_mean(entropy_cls, name='loss_cls') * factor
            # reg loss
            # pdb.set_trace()
            reg_loss_all = smooth_L1(self.tx - self.tx_star) + smooth_L1(self.ty - self.ty_star) + smooth_L1(self.tw - self.tw_star) + smooth_L1(self.th - self.th_star)
            positive_proposal_pixels = tf.where(self.proposal_scores < .5, tf.zeros_like(self.proposal_scores), tf.ones_like(self.proposal_scores))
            reg_loss_all = reg_loss_all * positive_proposal_pixels
            factor = 64 * self.batch_size / tf.reduce_sum(positive_proposal_pixels)
            self.loss_reg = tf.reduce_mean(reg_loss_all, name='loss_reg') * factor
            # RPN loss
            self.loss = tf.add(self.loss_cls, 10*self.loss_reg, name='loss')
            self.valid_pixels = valid_pixels
            
            # Faster RCNN classifier loss
            entropy_classifier = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.one_hot_label, logits=self.logits)
            self.loss_classifier = tf.reduce_mean(entropy_classifier)
            self.loss_fasterRCNN = self.loss_classifier + self.loss_cls + 10*self.loss_reg

            # Mask RCNN multi task losss
            entropy_mask = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.mask, logits=self.logits_mask)
            self.L_mask = tf.reduce_mean(entropy_mask, name='L_mask')
            self.mask_RCNN_loss = self.loss_fasterRCNN + self.L_mask

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.mask_RCNN_loss, 
                                                global_step=self.gstep)
    
    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss_cls_RPN', self.loss_cls)
            # tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('loss_reg', self.loss_reg)
            # tf.summary.scalar('accuracy_cls_RPN', 100*self.accuracy_cls)

            # tf.summary.scalar('loss_fasterRCNN', self.loss_fasterRCNN)
            tf.summary.scalar('loss_classifier', self.loss_classifier)
            # tf.summary.scalar('accuracy_classifier', 5 * self.accuracy_classifier)

            tf.summary.scalar('loss_mask', self.L_mask)
            tf.summary.scalar('mask_RCNN_loss', self.mask_RCNN_loss)
            tf.summary.scalar('accuracy_mask', 5 * self.accuracy_mask)
            tf.summary.scalar('IoU_mask', 5 * self.iou_mask)
            # tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    
    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds_cls = tf.where(self.scores_cls > 0.5, tf.ones_like(self.scores_cls), tf.zeros_like(self.scores_cls))
            correct_preds_cls = tf.equal(preds_cls, self.proposal_scores)
            valid_pixels = tf.where(self.proposal_scores < -.1, tf.zeros_like(self.proposal_scores), tf.ones_like(self.proposal_scores))
            self.accuracy_cls = tf.reduce_sum(tf.cast(correct_preds_cls, tf.float32)) / tf.reduce_sum(valid_pixels) 

            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1,output_type=tf.int32), tf.argmax(self.one_hot_label, 1,output_type=tf.int32))
            self.accuracy_classifier = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            
            preds_mask = tf.nn.sigmoid(self.logits_mask)
            preds_mask = tf.round(preds_mask)
            correct_preds_mask = tf.equal(tf.cast(preds_mask, tf.float32), self.mask)
            self.accuracy_mask = tf.reduce_sum(tf.cast(correct_preds_mask, tf.float32))/(22 * 22)
           
            self.iou_mask = iou(self.mask, preds_mask)
            self.iou_mask = tf.reduce_sum(self.iou_mask)

    
    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.proposal()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:

                _, l, summaries = sess.run([self.opt, self.mask_RCNN_loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
                # pdb.set_trace()
                # val, ind = sess.run([self.values, self.indices])
                # pdb.set_trace()

        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_layers/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step 
    
    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        total_correct_preds_mask = 0
        total_iou = 0
        try:
            while True:
               
                accuracy_batch, accuracy_mask_batch, iou_batch, summaries = sess.run([self.accuracy_classifier, self.accuracy_mask, self.iou_mask, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
                total_correct_preds_mask += accuracy_mask_batch
                total_iou += iou_batch
              
        except tf.errors.OutOfRangeError:
            pass
       
        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds *.5/self.n_test))
        print('Accuracy for mask at epoch {0}: {1} '.format(epoch, total_correct_preds_mask *.5/self.n_test))
        if(total_correct_preds * 0.5/self.n_test > .72):
            self.status = True
        print('IoU for mask at epoch {0}: {1} '.format(epoch, total_iou *.5/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    
    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir(path='checkpoints')
        utils.safe_mkdir(path='checkpoints/convnet_layers')
        writer = tf.summary.FileWriter('./graphs/convnet_layers', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_layers/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
                if(epoch > 25):
                    self.lr = .0001
                if(self.status):
                    break
        writer.close()

def smooth_L1(x):
    y = tf.where(tf.abs(x) < 1.0, 0.5 * x *x, abs(x) - .5)
    return y

def iou(true,pred):  #this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (tf.reduce_sum(intersection, axis=[1,2,3]) + 1e-9) / (tf.reduce_sum(union, axis=[1,2,3]) + 1e-9)

if __name__ == '__main__':
    model = RPN()
    model.build()
    model.train(n_epochs=90)