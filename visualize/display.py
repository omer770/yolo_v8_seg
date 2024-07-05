import json
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from shapely import geometry
from shapely.geometry import  MultiPolygon

def retrive_json_object(path):
  with open(path, 'r') as f:
    data = json.load(f)
  return data

def plt_poly(polygon,shape):
  fig, ax = plt.subplots()
  for poly in polygon.geoms:
      xe, ye = poly.exterior.xy
      #xe, ye = [1024,1024]
      ax.plot(xe, ye, color="blue")
  plt.ylim(0, shape[1])
  plt.xlim(0, shape[0])
  plt.show()
def get_multi_poly_fname(fname, pred_object,df_gt):
  classes = df_gt['Class Name'].unique()
  seg1_polylist =[]
  seg2_polylist =[]
  colors = ["green","blue"]
  c1 = []
  c2 = []
  for i in range(len(classes)):
    #c1[i] = colors[i]#colors[i]
    #c2[i] = 
    s = list(pred_object[fname].keys())
    for t in s:
      if pred_object[fname][t]['classname']==classes[i]:
        seg1_polylist.append(generate_poly_from_list(pred_object[fname][t]['segmentation']))
        c1.append(colors[i])
    sub_df = df_gt[(df_gt['File Name']== fname)&(df_gt['Class Name']== classes[i])]
    for seg in sub_df.Segmentation:
      seg2_polylist.append(generate_poly_from_list(seg))
      c2.append(colors[i])
  return seg1_polylist,seg2_polylist,c1,c2

def plt_multi_poly_lists(polygons1, polygons2,c1,c2,shape):
  fig, axs = plt.subplots(1, 2)
  for color,poly in zip(c1,MultiPolygon(polygons1).geoms ):
    #Mpoly = (poly)
    xe, ye = poly.exterior.xy
    axs[0].plot(xe, ye, color=color)
  axs[0].set_title('Predictions')
  axs[0].set_ylim(0, shape[1])
  axs[0].set_xlim(0, shape[0])

  for color,poly  in zip(c2,MultiPolygon(polygons2).geoms):
    #Mpoly = (poly) 
    xe, ye = poly.exterior.xy
    axs[1].plot(xe, ye, color=color)
  axs[1].set_title('Annotations')
  axs[1].set_ylim(0, shape[1])
  axs[1].set_xlim(0, shape[0])

  plt.show()

def plt_multi_poly(polygon,shape):
  fig, ax = plt.subplots()
  for poly in MultiPolygon(polygon).geoms:
      xe, ye = poly.exterior.xy
      #xe, ye = [1024,1024]
      ax.plot(xe, ye, color="blue")
  plt.ylim(0, shape[1])
  plt.xlim(0, shape[0])
  plt.show()
def save_PR_plot(save_path:str,aP_df_50_2_95:dict,verbose:bool=False):
  plot_path = os.path.join(save_path,'plots')
  os.makedirs(plot_path, exist_ok=True)
  for ap in aP_df_50_2_95.keys():
    for ap_cls in aP_df_50_2_95[ap].keys():
      if verbose: print(f"Class- {ap_cls}, Average Precision- {ap}:")
      df_ap_graph = aP_df_50_2_95[ap][ap_cls]
      if df_ap_graph is None:
        if verbose: print(f"No data points to plot @{ap}:{ap_cls}\n")
        continue
      ax = plt.gca()
      df_ap_graph.plot(kind='line',
                x='Recall_ap',
                y='Precision_ap',
                color='green', ax=ax)
      plt.ylabel('Precision')
      plt.xlabel('Recall')
      plt.title('PR curve for {} class with iou thrshold {}%'.format(ap_cls,ap))
      plt.savefig(os.path.join(plot_path,f"PR_curve_{ap}_{ap_cls}.jpg"))
      if verbose: print("Saved as "+os.path.join(plot_path,f"PR_curve_{ap}_{ap_cls}.jpg\n"))
      plt.clf()

def plot_kv_plot(save_path:str, dict_object:dict,title:str,x_label:str=None,y_label:str=None,show_maximum:bool=True,show_plot:bool=True):
  plot_path = os.path.join(save_path,'plots')
  os.makedirs(plot_path, exist_ok=True)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  x = list(dict_object.keys())
  y = list(dict_object.values())
  line, = ax.plot(x, y)
  ymax = max(y)
  xmax = x[y.index(ymax)]
  if show_maximum:
    ax.annotate('local max',xy = (xmax,ymax),xytext = (xmax,ymax+5),
                arrowprops = dict(facecolor = 'black',shrink = 0.05))
  if x_label and y_label:
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
  ax.set_title(title)
  ax.set_ylim(0,ymax*1.5)
  plt.xticks(rotation=45)
  plt.savefig(os.path.join(plot_path,f"{title}_{x_label}_{y_label}.jpg"))
  if show_plot:
    plt.show()

def plot_segmentations_from_json(image_name:str,image_dir_path:str,
                                 pred_json_path:str=None,label_json_path:str = None,
                                 pred_object:dict=None,gt_object:dict=None,
                                 save_path:str=None):
  """
  Plots the image with segmentation masks from labels.

  Args:
      image_path (str): Path to the image file.
      labels (list): A list of segmentation labels (see read_segmentation_labels).
  """

  if pred_json_path: pred_object = retrive_json_object(pred_json_path)
  if label_json_path: gt_object = retrive_json_object(label_json_path)
  
  image_path = os.path.join(image_dir_path,image_name)
  image = cv2.imread(image_path)
  height,width,ch = image.shape
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
  
  if gt_object:
    ext = list(gt_object.keys())[0].split('.')[-1]
    if ext == 'txt':label_name = image_name.replace('.jpg','.txt')
    else: label_name = image_name

  pred_data = [pred_object[image_name][i] for i in pred_object[image_name]]
  gt_data = [gt_object[label_name][i] for i in gt_object[label_name]] if gt_object else None
  pred_color = (0,0,255)
  gt_color = (0,255,0)
  isClosed = True
  thickness = 2
  fontScale = 1
  for obj in pred_data:
    polygon = [(x*width, y*height) for (x,y) in obj['segmentation']]
    polygon_points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    [image_x,image_y] = polygon_points.min(axis=0)[0].tolist()
    if image_x == 0 or image_y == 0:
      image_x , image_y  = image_x+50,image_y+50
    image = cv2.polylines(image, [polygon_points], isClosed, pred_color, thickness)
    cv2.putText(image, obj['classname'], (image_x,image_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, pred_color,thickness)
  if gt_object:
    for obj in gt_data:
      polygon = [(x*width, y*height) for (x,y) in obj['segmentation']]
      polygon_points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
      [label_x,label_y] = polygon_points.min(axis=0)[0].tolist()
      if label_x == 0 or label_y == 0:
        label_x , label_y  = label_x+50,label_y+50
      image = cv2.polylines(image, [polygon_points], isClosed, gt_color, thickness)
      cv2.putText(image, obj['classname'], (label_x,label_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, gt_color,thickness)
  plt.title(f"Detection plot of {image_name}", fontdict = {'fontsize' : 10})
  plt.axis('off')
  plt.imshow(image)
  if save_path: 
    plot_path = os.path.join(save_path,'plots')
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path,f"Detection_plot_{image_name}"))
  plt.show()

def save_pred_outline_plots(pred_json_path:str, save_path:str,image_dir:str,
                            classes:list,number_samples:int,show_plot:bool=False):
  pred_object = retrive_json_object(pred_json_path)
  colors_per_class = {class_name:tuple(np.random.choice(range(256), size=3).tolist()) for class_name in classes} # Convert NumPy integers to Python integers
  for s in range(number_samples):
    fname = np.random.choice(list(pred_object.keys()))
    image = cv2.imread(os.path.join(image_dir,fname))
    height,width,ch = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for id,obj in pred_object[fname].items():
      polygon = [(x*width, y*height) for (x,y) in obj['segmentation']]
      pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
      isClosed = True
      thickness = 2
      fontScale = 1
      color = colors_per_class[obj['classname']]
      image = cv2.polylines(image, [pts], isClosed, color, thickness)
      [image_x,image_y] = pts.min(axis=0)[0].tolist()
      if image_x == 0 or image_y == 0:
        image_x , image_y  = image_x+50,image_y+50
      cv2.putText(image, obj['classname'], (image_x,image_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color,thickness)
      
      saved_path = os.path.join(save_path,'saved_outline_plots')
      os.makedirs(saved_path, exist_ok=True)
      #plt.savefig(os.path.join(plot_path,f"Detection_plot_{image_name}"))
      if show_plot:
        plt.title(f"Prediction Outline of {fname}", fontdict = {'fontsize' : 10})
        plt.axis('off')
        plt.imshow(image)
        plt.show()
    cv2.imwrite(f'{saved_path}/image_{fname}',image)
    print('Saved: image_'+fname)
  print("Saved at location: ", saved_path)