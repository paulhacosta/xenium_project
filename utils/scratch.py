#%%
import tifffile as tiff
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
import matplotlib.patches as patches

#%%
image_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/enrichment/public_data/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/spatial_morph_proj_label_singleR.png"


# %%
img = Image.open(image_path)
crop_box = (300, 300, 2300, 1500)  # adjust as needed

fig, ax = plt.subplots()
ax.imshow(img)
rect = patches.Rectangle(
    (crop_box[0], crop_box[1]),
    crop_box[2] - crop_box[0],
    crop_box[3] - crop_box[1],
    linewidth=2,
    edgecolor='red',
    facecolor='none'
)
ax.add_patch(rect)
plt.title("Crop Preview")
plt.axis("off")
plt.show()

#%%
cropped = img.crop(crop_box)
fig, ax = plt.subplots()
ax.imshow(cropped)
plt.axis("off")
plt.show()

#%%
gray = cropped.convert("L")  # "L" mode = 8-bit pixels, black and white

array = np.array(gray)
tiff.imwrite("/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/enrichment/public_data/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/spatial_morph_proj_label_singleR_gray.tiff", array)

print("TIFF saved with integer pixel values.")
