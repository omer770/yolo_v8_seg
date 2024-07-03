import json
import os
import shutil
import yaml
from tqdm.auto import tqdm
from funcy import lmap, lfilter, lremove
from sklearn.model_selection import train_test_split

def split_coco_annotations(annotations_path, train_val_test_split: tuple = (0.75, 0.15, 0.1),
                           train_json_name="train.json", valid_json_name="valid.json", test_json_name="test.json",
                            filter_unannotated=False):
  """
  Splits COCO annotations into training, validation, and test sets.

  Args:
      annotations_path: Path to the COCO annotations file.
      train_val_test_split: 
        - Ratios of images for training,validation and test set (0.75, 0.1, 0.15).
        - It can be tuple of 0 to 3 elements.
        - If the tupple has 2 elements then the remaining or no data will be for test.
        - If the tupple has 1 element then the remaining will go for validation and no data will be for test.
      train_json_name: Filename to store training annotations (default: "train.json").
      valid_json_name: Filename to store validation annotations (default: "valid.json").
      test_json_name: Filename to store test annotations (default: "test.json").
      filter_unannotated: Whether to keep only images with annotations (default: False).

  Returns:
      tuple(train_json_name,valid_json_name,test_json_name)
  """
  if type(train_val_test_split) == float:
    len_tvt =  1
  else:
    len_tvt =  len(train_val_test_split)
  if len_tvt == 3:
    train_ratio, valid_ratio, test_ratio = train_val_test_split
    if train_ratio + valid_ratio + test_ratio != 1:
      raise ValueError("train_val_test_split must add up to 1.")
  elif len_tvt == 2:
    train_ratio, valid_ratio = train_val_test_split
    test_ratio = round(1 - train_ratio - valid_ratio,2)
    
  elif len_tvt == 1:
    train_ratio = train_val_test_split
    valid_ratio = round(1 - train_ratio,2)
    test_ratio = 0
  else:
    raise ValueError("train_val_test_split must be a tuple of 0 to 3 elements.")

  with open(annotations_path, 'rt', encoding='UTF-8') as annotations:
    coco = json.load(annotations)
    info = coco['info']
    licenses = coco['licenses']
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    number_of_images = len(images)

    if filter_unannotated:
      images_with_annotations = lmap(lambda a: int(a['image_id']), annotations)
      images = lremove(lambda i: i['id'] not in images_with_annotations, images)

    train_before, test = train_test_split(images, test_size=test_ratio)

    ratio_remaining = 1 - test_ratio
    ratio_valid_adjusted = valid_ratio / ratio_remaining

    train_after, valid = train_test_split(train_before, test_size=ratio_valid_adjusted)

    save_coco(train_json_name, info, licenses, train_after, filter_annotations(annotations, train_after), categories)
    save_coco(test_json_name, info, licenses, test, filter_annotations(annotations, test), categories)
    save_coco(valid_json_name, info, licenses, valid, filter_annotations(annotations, valid), categories)

    print("Saved {} entries in {} and {} in {} and {} in {}".format(len(train_after), train_json_name,len(valid), valid_json_name, len(test), test_json_name))
    return (train_json_name,valid_json_name,test_json_name)

def save_coco(file, info, licenses, images, annotations, categories):
  with open(file, 'wt', encoding='UTF-8') as coco:
    json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
        'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
  image_ids = lmap(lambda i: int(i['id']), images)
  return lfilter(lambda a: int(a['image_id']) in image_ids, annotations)
  
def convert_to_yolo(input_images_path:str, input_json_path:str, output_path:str,suffix_images_2_path:str=''):
    '''
    Convert COCO annotations to YOLOv8
    input_images_path: Path to the directory containing the images
    input_json_path: Path to the JSON file containing the annotations
    output_path: Path to the directory where the converted files will be saved
    eg: output_path ='/content/yolo_v8_seg/data/train'
    suffix_images_2_path: Path to the directory containing the images (optional)

    '''
    # Open JSON file containing image annotations

    with open(input_json_path, 'rt', encoding='UTF-8') as f:
      data = json.load(f)
    output_images_path = os.path.join(output_path, 'images')
    output_labels_path = os.path.join(output_path, 'labels')

    # Create directories for output images and labels
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)
    # Function to get image annotations
    def get_img_ann(image_id):
        return [ann for ann in data['annotations'] if ann['image_id'] == image_id]

    # Function to get image data
    def get_img(filename):
        return next((img for img in data['images'] if img['file_name'] == filename), None)
    # List to store filenames
    file_names = []
    for filename in tqdm(os.listdir(input_images_path),desc="Scanning files"):
        if filename.endswith(".jpg"):
            source = os.path.join(input_images_path, filename)
            destination = os.path.join(output_images_path, filename)
            if get_img(os.path.join(suffix_images_2_path,filename)):
              shutil.copy(source, destination)
              file_names.append(filename)
    assert len(file_names) > 0, f"No images found in {input_images_path},Use 'suffix_images_2_path' for adding suffix for search"
    # Iterate through filenames and process each image
    for filename in tqdm(file_names,desc="Processing files"):
        img = get_img(os.path.join(suffix_images_2_path,filename))
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        img_ann = get_img_ann(img_id)

        # Write normalized polygon data to a text file
        if img_ann:
            with open(os.path.join(output_labels_path, f"{os.path.splitext(filename)[0]}.txt"), "a") as file_object:
                for ann in img_ann:
                    current_category = ann['category_id'] - 1
                    polygon = ann['segmentation'][0]
                    normalized_polygon = [format(coord / img_w if i % 2 == 0 else coord / img_h, '.6f') for i, coord in enumerate(polygon)]
                    file_object.write(f"{current_category} " + " ".join(normalized_polygon) + "\n")
    print(f"Converted {len(file_names)} images to YOLOv8 format")
    return output_path
# Function to create a YAML file for the dataset
def create_yaml(input_json_path, output_yaml_path, train_path, val_path, test_path=None):
    with open(input_json_path) as f:
        data = json.load(f)
    
    # Extract the category names
    names = [category['name'] for category in data['categories']]
    
    # Number of classes
    nc = len(names)

    # Create a dictionary with the required content
    yaml_data = {
        'names': names,
        'nc': nc,
        'test': test_path if test_path else '',
        'train': train_path,
        'val': val_path
    }

    # Write the dictionary to a YAML file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

    print(f"YAML file created at {output_yaml_path}")
    return output_yaml_path

if __name__ == "__main__":
    base_input_path = "coco_dataset/"
    base_output_path = "yolo_dataset/"
    # Processing validation dataset (if needed)
    convert_to_yolo(
        input_images_path=os.path.join(base_input_path, "data/images/val"),
        input_json_path=os.path.join(base_input_path, "data/annotations/val.json"),
        output_path=os.path.join(base_output_path, "valid")
    )

    # Processing training dataset 
    convert_to_yolo(
        input_images_path=os.path.join(base_input_path, "data/images/test"),
        input_json_path=os.path.join(base_input_path, "data/annotations/test.json"),
        output_path=os.path.join(base_output_path, "test")
    )
    
    # Creating the YAML configuration file
    create_yaml(
        input_json_path=os.path.join(base_input_path, "data/annotations/train.json"),
        output_yaml_path=os.path.join(base_output_path, "data.yaml"),
        train_path=base_output_path+"/data/images/train",
        val_path=base_output_path+"data/images/val",
        test_path= None#'../test/images'  # or None if not applicable
    )
