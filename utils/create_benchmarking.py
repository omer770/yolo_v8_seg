import os
import json
import yaml
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

def read_segmentation_labels(label_path:Path, classes:list):
  label_cords = {}
  with open(label_path, 'r') as f:
    count = 0
    for line in f.readlines():
      # Assuming each line represents a polygon
      data = line.strip().split(' ')
      polygon_coords = []
      for i in range(1, len(data), 2):  # Extract coordinates in pairs
        x, y = float(data[i]), float(data[i+1])
        polygon_coords.append((x, y))
      polygon_coords = np.array(polygon_coords)
      label_cords[count] =  {'classname':classes[int(data[0])],'segmentation': polygon_coords}
      count += 1
  return label_cords

def generate_benchmarking_json(model,test_dir_path:Path, yaml_path:Path = None , is_label_present=None):

  test_dir_items = os.listdir(test_dir_path)
  dict_i, dict_a= {},{}
  #annotations
  if is_label_present:
    assert yaml_path , "Provide path to yaml using 'yaml_path'"
    test_label_path = test_dir_path.parent/'labels'
    test_label_items = os.listdir(test_label_path)
    
    with open(yaml_path, 'r') as stream:
      classes = yaml.safe_load(stream)['names']
    for i in tqdm(range(len(test_label_items))):
      label_dicts = read_segmentation_labels(test_label_path/test_label_items[i],classes)
      dict_a[test_label_items[i]] = label_dicts
  #Predictions
  for i in tqdm(range(len(test_dir_items))):
    dict_j = {}
    predicts = model.predict(os.path.join(test_dir_path,test_dir_items[i]) , conf=0.3)
    bbox = predicts[0].boxes.data[:,:-1]
    cls_id = predicts[0].boxes.data[:,-1]
    cls_rows = [list(predicts[0].names.values())[int(x)] for x in cls_id ]
    confidence = list(predicts[0].boxes.conf.cpu().numpy())
    for j in range(len(cls_rows)):
      dict_j[j] = {'classname':cls_rows[j],'confidence': confidence[j],'segmentation': predicts[0].masks.xyn[j]}
    dict_i[test_dir_items[i]] = dict_j
  return dict_i, dict_a 