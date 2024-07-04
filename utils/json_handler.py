import json
import os
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        return json.JSONEncoder.default(self, obj)

def print_structure(d, indent=0):
    """Prints the structure of a dictionary or list."""

    # If the input is a dictionary
    if isinstance(d, dict):
        for key, value in d.items():
            print('  ' * indent + str(key))
            print_structure(value, indent+1)

    # If the input is a list
    elif isinstance(d, list):
        print('  ' * indent + "[List of length {} containing:]".format(len(d)))
        if d:
            print_structure(d[0], indent+1)
    else:
        print('  ' * indent + str(type(d)).replace('<','').replace('>','').replace(' ',':'))
          
def retrive_json_object(json_file):
  assert json_file.split('.')[-1] == 'json' , f"Provide a Json extention filename, provided-'{json_file}' "
  # Open the JSON file and read its content
  with open(json_file, 'r') as f:
    json_string = f.read()
  
  # Decode the JSON string into a Python object
  json_object = json.loads(json_string)
  return json_object

def export_json_file(json_object,filename):
  assert filename.split('.')[-1] == 'json' , f"Provide a Json extention filename, provided-'{filename}' "
  if os.path.exists(filename):
    os.remove(filename)
  # Open the file and read its content
  dumped = json.dumps(json_object, cls=NumpyEncoder)
  with open(filename, 'a') as f:
      f.write(dumped + '\n')
  print("Saved- ",filename)
  return filename
  
