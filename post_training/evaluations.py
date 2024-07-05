import json
import copy
import numpy as np
import pandas as pd
from shapely import geometry
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from typing import List, Dict, Tuple, Any
pd.options.mode.chained_assignment = None


def get_classes(data_yaml:str):
  import yaml
  with open(data_yaml, 'r') as stream:
    data = yaml.safe_load(stream)
    classes = data['names']
    return classes

def is_valid_segmentation(segmentation)->bool:
  """
  Checks if a segmentation mask (represented as a list of coordinates) is valid using shapely.

  Args:
      segmentation: A list of coordinates representing the segmentation mask.

  Returns:
      True if the segmentation is valid, False otherwise.
  """
  try:
    poly = Polygon(segmentation)
    return poly.is_valid
  except:
    return False

def generate_poly_from_list(segmentation_list: str | list) -> Polygon:
  """Generates a Shapely Polygon object from a list of coordinates.

  Args:
      segmentation_list: A list of coordinates or a string representation of a list of coordinates.

  Returns:
      A Shapely Polygon object.
  """

  if isinstance(segmentation_list, str):
    segmentation_list = segmentation_list.replace('[', '').replace(']', '').split(',')
    segmentation_list = [[int(segmentation_list[p]), int(segmentation_list[p + 1])] for p in range(0, len(segmentation_list), 2)]

  if segmentation_list[0] != segmentation_list[-1]:
    if abs(segmentation_list[0][0] - segmentation_list[-1][0]) <= 7 and abs(segmentation_list[0][1] - segmentation_list[-1][1]) <= 7:
      segmentation_list[-1] = segmentation_list[0]
    else:
      segmentation_list.append(segmentation_list[0])

  return Polygon(segmentation_list)

def calculate_iou(seg1: List[float], seg2: List[float]) -> float:
    """Calculates the Intersection over Union (IoU) between two segmentation masks.

    Args:
        seg1: The first segmentation mask as a list of coordinates.
        seg2: The second segmentation mask as a list of coordinates.

    Returns:
        The IoU value between the two masks.
    """
    poly1 = generate_poly_from_list(seg1)
    poly2 = generate_poly_from_list(seg2)
    return poly1.intersection(poly2).area / poly1.union(poly2).area


def map_calculate_per_iou(pred_object,gt_object,classes,iou_threshold,conf_threshold,verbose:bool=False):
  tp_fp,len_gt,polygons1,polygons =[],[],[],[]
  fname_labels = list(gt_object.keys())
  FileName_list,Object_list,Confidence_list,TP_list,FP_list,Class_list = [],[],[],[],[],[]
  df_ap_class = {}
  for fname_label in fname_labels:
    fname_img= fname_label.replace('.txt','.jpg')
    count_gt_class = 0
    if verbose:
      print(f"File Name- {fname_img}, iou_threshold- {iou_threshold}")
      print("-"*85)
    if len(pred_object[fname_img])==0:continue
    tp_fp_1, len_gt1 =[],[]
    for cls in classes:
      if verbose: print("Class- ",cls)
      id_preds = list(pred_object[fname_img].keys())
      t_f =[]
      len_gt2 = []
      count_np = 0
      sub_dict_gt = {id_item: dict_item for id_item,dict_item in test_dict[fname_label].items() if dict_item['classname'] == cls}
      for id_pred in id_preds:
        if pred_object[fname_img][id_pred]['confidence'] < conf_threshold:  continue
        seg1 = pred_object[fname_img][id_pred]['segmentation']
        if pred_object[fname_img][id_pred]['classname'] != cls:
          count_np+=1
          continue
        for id_gt in sub_dict_gt.keys():
          seg2 = sub_dict_gt[id_gt]['segmentation']
          try:iou = calculate_iou(seg1,seg2)
          except:
            polygons.append([generate_poly_from_list(seg1),generate_poly_from_list(seg2)])
            iou = 0.1
          if iou >= iou_threshold:
            t_f.append(1)
            FileName_list.append(fname_img)
            Object_list.append(id_pred)
            Confidence_list.append(pred_object[fname_img][id_pred]['confidence'])
            Class_list.append(cls)
            TP_list.append(1)
            FP_list.append(0)
            break
          else:
            if (id_gt==list(sub_dict_gt.keys())[-1]):
              t_f.append(0)
              FileName_list.append(fname_img)
              Object_list.append(id_pred)
              Confidence_list.append(pred_object[fname_img][id_pred]['confidence'])
              TP_list.append(0)
              FP_list.append(1)
              Class_list.append(cls)
              polygons1.append([generate_poly_from_list(seg1),generate_poly_from_list(seg2)])
        len_gt2.append(len(sub_dict_gt))
        if np.sum(t_f) == len(sub_dict_gt):break
      if verbose: print("t_f, sum, len ",t_f,np.sum(t_f),len(t_f))
      if len(t_f)!=0:
        pre = (np.sum(t_f))/(len(t_f)) if (len(t_f))!= 0 else 0.0001
        rec = (np.sum(t_f))/(len(sub_dict_gt)) if (len(sub_dict_gt))!=0 else 0.0001
        if verbose:
          print("len pred ",len(id_preds))
          print(f"len of gt {len(sub_dict_gt)}\n")
          print("Precision and Recall for class- ",cls,pre,rec)
      tp_fp_1.append(t_f)
      len_gt1.append(len(sub_dict_gt))
      if verbose:
        print("len pred ",len(id_preds)-count_np)
        print(f"len of gt {len(sub_dict_gt)}\n")
        print("-"*85)
    tp_fp.append(tp_fp_1)
    len_gt.append(len_gt1)
  df_ap = pd.DataFrame(zip(FileName_list,Object_list,Confidence_list,Class_list,TP_list,FP_list),columns = ["Image","Detection", "Confidence", "Class","TP","FP"])
  df_ap = df_ap.sort_values(by=['Confidence'],ascending=False)
  for cls in classes:
    df_ap_class[cls] = df_ap[df_ap['Class'] == cls]
    df_apc = df_ap_class[cls]
    df_apc = perform_pr_process(df_apc)

  return tp_fp,len_gt,df_ap_class

def overall_PR(tp_fp,len_gt,classes):
  pc = 0
  rc = 0
  count = 0
  print("-"*43)
  count_ngt = 0
  #classes = ['building','pool']
  for c in [0,1]:
    prc = 0
    rcc = 0
    count_ngt1 = 0
    for i in range(len(len_gt)):
      pri, rci = precision_recall_per_object(tp_fp,len_gt, i,c)
      if (pri, rci)==(0.0001, 0.0001):
        count_ngt1+=1
      prc += pri
      rcc += rci
    #print("nt1",count_ngt1)
    prcn = prc /(len(tp_fp)-count_ngt1)
    rccn = prc /(len(len_gt)-count_ngt1)

    count_ngt+=count_ngt1
    print("precision for ",classes[c],prcn)
    print("recall for ",classes[c],rccn)
    print("-"*43)
    pc+=prc
    rc+=rcc
  pc/=(len(tp_fp)*2-count_ngt)
  rc/=(len(tp_fp)*2-count_ngt)
  print("-"*43)
  print("The overall precision is ",pc)
  print("The overall recall is ",rc)
  print("-"*43)
  return None

def precision_recall_per_object(tp_fp,len_gt, idr = 0,clas = 0):
  #file_index
  # 1 for pool, 0 for building
  precision = (np.sum(tp_fp[idr][clas]))/(len(tp_fp[idr][clas])) if (len(tp_fp[idr][clas])) != 0 else 0.0001
  recall = (np.sum(tp_fp[idr][clas]))/(len_gt[idr][clas]) if (len_gt[idr][clas]) != 0 else 0.0001
  return precision,recall

def perform_pr_process(df_apc):
  Acc_TP = []
  Acc_FP = []
  Precision_ap = []
  Recall_ap = []
  count_acc_tp =  0
  count_acc_fp = 0
  #df_apc = df_apc.sort_values(by=['Confidence'])
  len_gt_subclass = len(df_apc)
  for row_T,row_F in df_apc.loc[:,["TP","FP"]].values:
    count_acc_tp += row_T
    count_acc_fp += row_F
    Acc_TP.append(count_acc_tp)
    Acc_FP.append(count_acc_fp)
    presn = (count_acc_tp)/(count_acc_tp+count_acc_fp) if (count_acc_tp+count_acc_fp)!= 0 else 0
    rcll = (count_acc_tp)/(len_gt_subclass) if len_gt_subclass!= 0 else 0
    Precision_ap.append(presn)
    Recall_ap.append(rcll)
  df_apc["Acc_TP"] = np.array(Acc_TP)
  df_apc["Acc_FP"] = np.array(Acc_FP)
  df_apc["Precision_ap"] = np.array(Precision_ap)
  df_apc["Recall_ap"] = np.array(Recall_ap)
  df_apc["F1_score"] = 2 * ((df_apc["Precision_ap"] * df_apc["Recall_ap"]) / (df_apc["Precision_ap"] + df_apc["Recall_ap"]))
  df_apc = df_apc.sort_values(by=['F1_score'],ascending=False)
  df_apc = df_apc.reset_index(drop=True)
  return df_apc

def calculate_maP(df_ap_class, classes, iou_threshold,verbose:bool=False,warn_empty_class:bool = False):
  """
  This function calculates the maP score for a given IoU threshold.
  """
  aP = {}
  maP = 0
  ap_df = {}
  for cls in classes:
    if cls in df_ap_class.keys():
      df_ap_graph = df_ap_class[cls]
      # Check if there are enough data points to calculate AUC
      if df_ap_graph.shape[0] >= 2:
        aP[cls] = round(auc(df_ap_graph['Recall_ap'],df_ap_graph['Precision_ap']),3)
        ap_df[cls] = df_ap_graph.loc[:,['Recall_ap','Precision_ap']]
      else:
        if warn_empty_class: print(f"Warning: Not enough data points to calculate AUC for class {cls}")
        aP[cls] = 0  # Or handle this case differently as needed
        ap_df[cls] = None
  if aP:  # Check if aP dictionary is not empty
    maP = round(sum(aP.values())/len(aP),3)
  if verbose:
    print("The iou threshold of the model ",iou_threshold)
    print("The aP of the model ",aP)
    print("The maP of the model ",maP,'\n')
  return maP,aP,ap_df


