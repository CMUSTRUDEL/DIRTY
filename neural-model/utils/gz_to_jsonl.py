import tarfile
import sys
import json
from tqdm import tqdm

file_name = sys.argv[1]
tgt_file_name = sys.argv[2]

with open(tgt_file_name, 'w') as f_tgt:
    with tarfile.open(file_name, 'r') as f:
        for filename in tqdm(f.getmembers()):
            json_file = f.extractfile(filename)
            if json_file is not None:
                json_str = json_file.read()
                json_str = json.dumps(json.loads(json_str))
                f_tgt.write(json_str + '\n')
