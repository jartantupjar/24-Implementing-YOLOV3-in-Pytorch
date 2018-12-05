from __future__ import division 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import predict_transform,get_device
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import bbox_iou 
def get_test_input():
    img = cv2.imread("images/dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    img_=img_.to(get_device())
    
    return img_

def parse_cfg(path):
 
    lines= open(path,'r').read().split('\n')
    #remove cfg comments and strip empty spaces
    lines=[x.rstrip().lstrip() for x in lines if x and not x.startswith('#')]
    
    modules=[]
    for line in lines:
        #insert blank dict
        if line.startswith('['):
           
            modules.append({})
            modules[-1]['type'] = line[1:-1].rstrip()
        
        #fill blank dict
        else:
            key,val=line.split('=')
            modules[-1][key.rstrip()]=val.lstrip()

    return modules

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()
    

        
           
def create_modules(parsed_modules):
    module_list=nn.ModuleList()
    net_params=parsed_modules.pop(0)
    prev_filters=int(net_params['channels'])
    output_filters=[]
    
    #loop through all parsed blocks(except 'net')
    for idx,line in enumerate(parsed_modules):
        x=line
        index=idx
        
        module=nn.Sequential()
  
        line_type=line['type']
        
        if line_type=='convolutional':
           
            if 'batch_normalize' in line:
                bn=int(line['batch_normalize'])
                bias=False
            else:
                bn=0
                bias=True
                
            filters= int(line["filters"])
            kernel_size = int(line["size"])
            stride = int(line["stride"])
            padding=(kernel_size-1)//2 if int(line['pad']) else 0
            
            module.add_module('conv_{0}'.format(idx), nn.Conv2d(
                                                        in_channels=prev_filters,
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=stride,
                                                        padding=padding,
                                                        bias=bias))
            
            if bn:
                module.add_module('batch_norm_{0}'.format(idx),nn.BatchNorm2d(filters))
                
            if line['activation']=='leaky':
             #check if we want to keep the inplace.
                module.add_module('leaky_{0}'.format(idx),nn.LeakyReLU(0.1,inplace=True))
                
        elif line_type=='shortcut':
            #filters=output_filters[int(line['from'])]
            
            module.add_module('shortcut_{0}'.format(idx),EmptyLayer())
                          
        elif line_type=='route':
            #recheck ##could be unstable or wrong
            layers=[int(layer) for layer in line['layers'].split(',')]
            
            module.add_module('route_{0}'.format(idx),EmptyLayer())
            
            filters=sum([output_filters[layer] for layer in layers])
            
        elif line_type=='yolo':    
            mask=[int(mask) for mask in line['mask'].split(',')]
            anchor=[int(anchor) for anchor in line['anchors'].split(',')]
            anchor=[(anchor[i],anchor[i+1]) for i in range(0,len(anchor),2) ]
            anchor=[anchor[i] for i in mask]
            num_classes = int(line['classes'])
            img_height = int(net_params['height'])
            
            module.add_module('yolo_{0}'.format(idx),DetectionLayer(anchor, num_classes, img_height))
              
        elif line_type=='upsample':
             module.add_module('upsample_{0}'.format(idx),nn.Upsample(
                 #            scale_factor=2,
                  scale_factor=int(line['stride']),
                             mode='nearest'))

        
        else:
            print('unknown layer')
            
        module_list.append(module)
        prev_filters=filters
        output_filters.append(filters)
        
    return(net_params,module_list)

import math
def get_targets(pred_boxes,pred_conf,pred_cls,targets,anchors,num_anchors,num_classes,in_h,ignore_thres,img_dim):
    
    mask=torch.zeros(targets.size(0),num_anchors,in_h,in_h)
    conf_mask=torch.zeros(targets.size(0),num_anchors,in_h,in_h)
    tx=torch.zeros(targets.size(0),num_anchors,in_h,in_h)
    ty=torch.zeros(targets.size(0),num_anchors,in_h,in_h)
    tw=torch.zeros(targets.size(0),num_anchors,in_h,in_h)
    th=torch.zeros(targets.size(0),num_anchors,in_h,in_h)
    tconf=torch.ByteTensor(targets.size(0),num_anchors,in_h,in_h).fill_(0)
    tcls=torch.ByteTensor(targets.size(0),num_anchors,in_h,in_h,num_classes).fill_(0)

    counter=0
    correct=0
    
    for batch in range(targets.size(0)):
        for t in range(targets.shape[1]):
            if targets[batch,t].sum()==0:
                continue
            counter+=1
            
            gx=targets[batch,t,1]*in_h
            gy=targets[batch,t,2]*in_h
            gw=targets[batch,t,3]*in_h
            gh=targets[batch,t,4]*in_h
            
            gi=int(gx)
            gj=int(gy)
            
            gt_box=torch.FloatTensor(np.array([0,0,gw,gh])).unsqueeze(0)
            anchor_shapes=torch.FloatTensor(np.concatenate((np.zeros((num_anchors,2)),np.array(anchors)),1))
            anch_ious=bbox_iou(gt_box,anchor_shapes,True)
            conf_mask[batch,anch_ious>ignore_thres,gj,gi]=0
            best_n=np.argmax(anch_ious)
            gt_box=torch.FloatTensor(np.array([gx,gy,gw,gh])).unsqueeze(0)
            pred_box=pred_boxes[batch,best_n,gj,gi].unsqueeze(0)
            mask[batch,best_n,gj,gi]=1
            conf_mask[batch,best_n,gj,gi]=1
            
            tx[batch,best_n,gj,gi]=gx-gi
            ty[batch,best_n,gj,gi]=gy-gj
            
            tw[batch,best_n,gj,gi]=math.log(gw/anchors[best_n][0]+1e-16)
            th[batch,best_n,gj,gi]=math.log(gh/anchors[best_n][1]+1e-16)
            
            target_label=int(targets[batch,t,0])
            tcls[batch,best_n,gj,gi,target_label]=1
            tconf[batch,best_n,gj,gi]=1
            
            iou=bbox_iou(gt_box,pred_box)
            pred_label=torch.argmax(pred_cls[batch,best_n,gj,gi])
            score=pred_conf[batch,best_n,gj,gi]
            if iou>0.5 and pred_label==target_label and score>0.5:
                correct+=1
    return counter,correct,mask,conf_mask,tx,ty,tw,th,tconf,tcls
            
            
            
from collections import defaultdict            
    

class DetectionLayer(nn.Module):
    def __init__(self,anchor,num_classes, img_dim):
   #def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors=anchor
        self.num_classes=num_classes
        self.bbox_attrs=5+num_classes
        self.num_anchors=len(self.anchors)
        self.img_dim=img_dim
        self.ignore_thres=0.5
        self.lambda_coord=1
        
        self.mse_loss=nn.MSELoss()
        self.bce_loss=nn.BCELoss()
        self.ce_loss=nn.CrossEntropyLoss()
        
    
    def forward(self,x,targets=None):
        bs = x.size(0)
        in_h = x.size(2)
        stride= self.img_dim / in_h
        
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        
        prediction = x.view(bs,  self.num_anchors,self.bbox_attrs, in_h, in_h).permute(0, 1, 3, 4, 2).contiguous()
        
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        pred_conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        
        
        # Calculate offsets for each grid
        grid_x = torch.arange(in_h).repeat(in_h, 1).view([1, 1, in_h, in_h]).type(FloatTensor)
        grid_y = torch.arange(in_h).repeat(in_h,1).t().view([1,1,in_h,in_h]).type(FloatTensor)
        # Calculate anchor w, h
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])    
        anchor_w=scaled_anchors[:,0:1].view((1,self.num_anchors,1,1))
        anchor_h=scaled_anchors[:,1:2].view((1,self.num_anchors,1,1))
            # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h        
        
        
        if targets is None:
           
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride,
                                pred_conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output 
        else:
            
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor  
            
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()
            
            counter,correct,mask,conf_mask,tx,ty,tw,th,tconf,tcls=get_targets(
                pred_boxes.cpu().data,pred_conf.cpu().data,pred_cls.cpu().data,targets.cpu().data,
                scaled_anchors.cpu().data,self.num_anchors,self.num_classes,in_h,
                self.ignore_thres,self.img_dim)
            
            total=int((pred_conf>0.5).sum().item())
            recall=float(correct/counter) if counter else 1
            precision=float(correct/total)
            
            mask=Variable(mask.type(ByteTensor))
            conf_mask=Variable(conf_mask.type(ByteTensor))
            
            tx=Variable(tx.type(FloatTensor),requires_grad=False)
            ty=Variable(ty.type(FloatTensor),requires_grad=False)
            tw=Variable(tw.type(FloatTensor),requires_grad=False)
            th=Variable(th.type(FloatTensor),requires_grad=False)
            tconf=Variable(tconf.type(FloatTensor),requires_grad=False)
            tcls=Variable(tcls.type(FloatTensor),requires_grad=False)
            
            conf_mask_true=mask
            conf_mask_false=conf_mask-mask
            
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            loss_cls=(1/bs)*self.ce_loss(pred_cls[mask],torch.argmax(tcls[mask],1))
            loss=loss_x+loss_y+loss_w+loss_h+loss_conf+loss_cls

            return (loss,loss_x.item(),loss_y.item(),loss_w.item(),loss_h.item(),
                    loss_conf.item(),loss_cls.item(),recall,precision)

    
class Darknet(nn.Module):
    def __init__(self,cfg_path):
        super(Darknet,self).__init__()
        self.parsed_modules=parse_cfg(cfg_path)
        self.net_params,self.module_list=create_modules(self.parsed_modules)
        #self.img_size=img_size      
        self.header= torch.IntTensor([0,0,0,0])
        self.seen=0
        #self.header= np.array([0,0,self.seen,0])
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall','precision']
        
        
    def forward(self,x,targets=None):
        is_training= targets is not None
        self.losses=defaultdict(float)
        #is_training=False

        detections=[]
        detected=False
        outputs = []
        for idx,module in enumerate(self.parsed_modules):
            module_type=(module['type'])
            
            if module_type in ['convolutional','upsample', 'maxpool']:
                x=self.module_list[idx](x)
                
            elif module_type=='route':
                layers=[int(layer) for layer in module['layers'].split(',')]
                x=torch.cat([outputs[i] for i in layers],1)
                
            elif module_type=='shortcut':
                layers=int(module['from'])
                
                x=outputs[-1]+outputs[idx+layers]
                
            elif module_type=='yolo':
                if not is_training:
                    x=self.module_list[idx](x)
                    if type(x)==int:
                        continue
                    if detected:
                        detections=torch.cat((detections,x),1)
                    else:
                        detections=x
                        detected=True
                        
                else:
                    #print(targets.shape)
                    x,*losses=self.module_list[idx][0](x,targets)
                    for name,loss in zip(self.loss_names,losses):
                        self.losses[name]+=loss
                    detections.append(x)
                        
                    
            outputs.append(x)    
        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        return sum(detections) if is_training else detections                     

    
    def load_weights(self,weights_path):
            fp=open(weights_path,"rb")
            
            self.header= torch.from_numpy(np.fromfile(fp,dtype=np.int32,count=5))
            self.seen=self.header[3]
            weights=np.fromfile(fp,dtype=np.float32)
            
            
            ptr=0
            for idx,module in enumerate(self.parsed_modules):
                module_type=module['type']
                
                conv=self.module_list[idx][0]
                
                if module_type=='convolutional':
                    if 'batch_normalize' in module:
                        
                        bn=self.module_list[idx][1]
                        
                        num_bias=bn.bias.numel()
                        
                        bn_biases=torch.from_numpy(weights[ptr:ptr+num_bias])
                        ptr+=num_bias
                        bn_weights=torch.from_numpy(weights[ptr:ptr+num_bias])
                        ptr+=num_bias
                        bn_running_mean=torch.from_numpy(weights[ptr:ptr+num_bias])
                        ptr+=num_bias
                        bn_running_var=torch.from_numpy(weights[ptr:ptr+num_bias])
                        ptr+=num_bias
                        
                        bn_biases=bn_biases.view_as(bn.bias.data)
                        bn_weights=bn_weights.view_as(bn.weight.data)
                        bn_running_mean=bn_running_mean.view_as(bn.running_mean)
                        bn_running_var=bn_running_var.view_as(bn.running_var)
                        
                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)
                        
                    else:
                        
                        num_bias=conv.bias.numel()
                        conv_biases=torch.from_numpy(weights[ptr:ptr+num_bias])
                        ptr+=num_bias
                        
                        conv_biases=conv_biases.view_as(conv.bias.data)
                        conv.bias.data.copy_(conv_biases)
                        
                    num_weights=conv.weight.numel()     
                    
                    conv_weights=torch.from_numpy(weights[ptr:ptr+num_weights])
                    ptr+=num_weights
                    
                    conv_weights=conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)
                             
                             
    
    
    
            
            
                
            
            

