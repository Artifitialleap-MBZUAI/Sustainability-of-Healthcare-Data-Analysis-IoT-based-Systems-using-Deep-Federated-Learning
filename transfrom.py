import pickle
import os
from PIL import Image
import numpy as np

output = open('data_atlas.pkl', 'ab')


mainfolder="/Atlas-images/"



co= os.listdir(mainfolder)
for folder in co:
      img = cv2.imread(os.path.join(mainfolder,folder))
      if img is not None:
        img = img.resize((224, 224))
        img = np.array(img)
        categ= re.sub('\d+(.jpeg)','',folder)
        #norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img.astype(np.uint8)
        #output = open('data.pkl', 'ab')
        pickle.dump({"img":img, "img_name":folder, "category":categ}, output)
        #output.close()

