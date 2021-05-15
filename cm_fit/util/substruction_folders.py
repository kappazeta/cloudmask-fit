import os
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score

# Take product name # Experiment path
# Output big original image, prediction, labels, sen2cor

# parse product name
# initialize big image
# loop through numbers
# PRODUCT = "S2A_MSIL2A_20200529T094041_N0214_R036_T35VLF_20200529T120441.CVAT"
# PRODUCT = "S2A_MSIL2A_20200509T094041_N0214_R036_T35VME_20200509T111504.CVAT"
PRODUCT = "S2B_MSIL2A_20200401T093029_N0214_R136_T34UFA_20200401T122148.CVAT"
EXPERIMENT_PATH = "/sar/data/cvat/code/cloudmask-fit/results/finetune_june"
TILESIZE = 512
NAMESIZE = 1


if __name__ == "__main__":
    # Parsing Product name
    file_specificator = PRODUCT.rsplit('.', 1)[0]
    date_match = file_specificator.rsplit('_')[2]
    index_match = file_specificator.rsplit('_', 1)[0].rsplit('_', 1)[-1]

    folder1 = EXPERIMENT_PATH + "/validation"
    folder2 = EXPERIMENT_PATH + "/validation_B06"
    names = ["/validation_B08", "/validation_B8A", "/validation_B09", "/validation_B11", "/validation_B12", "/validation_WVP"]
    for name in names:
        folder2 = EXPERIMENT_PATH + name
        set_x = set()
        set_y = set()
        classes_importance = {66:0,129:0,192:0,255:0}
        classes_list = [66, 129, 192, 255]
        counter = {66:0, 129:0, 192:0, 255:0}

        for subfolder in os.listdir(folder1):
            if subfolder.startswith(index_match + "_" + date_match):
                pred_im_real = np.asarray(Image.open(folder1 + "/" + subfolder + "/prediction.png"))
                pred_im_2 = np.asarray(Image.open(folder2 + "/" + subfolder + "/prediction.png"))
                real_curr = np.zeros_like(pred_im_real)
                real_compare = np.zeros_like(pred_im_real)
                for i in classes_list:
                    real_curr[pred_im_real == i] = pred_im_real[pred_im_real == i]
                    real_compare[pred_im_2 == i] = pred_im_2[pred_im_2 == i]
                    real_curr[real_curr > 0] = 1
                    real_compare[real_compare > 0] = 1
                    real_curr = real_curr.flatten()
                    real_compare = real_compare.flatten()
                    curr_f1_score = f1_score(real_curr, real_compare)
                    classes_importance[i] += curr_f1_score

                    if real_curr.sum() > 0:
                        counter[i] += 1
                    real_curr = np.zeros_like(pred_im_real)
                    real_compare = np.zeros_like(pred_im_real)

        for key, value in classes_importance.items():
            classes_importance[key] = value/counter[key]

        print(classes_importance, counter)