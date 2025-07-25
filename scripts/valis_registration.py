#%%
import os, sys

sys.path.insert(0, "/rsrch5/home/plm/phacosta/xenium_project/valis")
from valis import registration, slide_io, non_rigid_registrars
from pathlib import Path
from valis.micro_rigid_registrar import MicroRigidRegistrar # For high resolution rigid registration
import time 
import numpy as np
#%%

root_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/TMP-IL-Pilot/20250515__183240__CIMAC_Validation/test_registration"

slide_src_dir = os.path.join(root_dir, "slides")
results_dst_dir = os.path.join(root_dir, "slide_registration_example_v3")
registered_slide_dst_dir = os.path.join(root_dir, "slide_registration_example_v3/registered_slides")
reference_slide = "morphology_focus_0000_converted.ome.tif"
reference_slide_fullpath = os.path.join(slide_src_dir, reference_slide)
img_to_register = "Xenium H&E Meso1-ICON2 TMA 5-21-2025_matching_orientation.ome.tif"
img_to_register_fullpath = os.path.join(slide_src_dir, img_to_register)

assert os.path.exists(reference_slide_fullpath)
assert os.path.exists(img_to_register_fullpath)

micro_reg_fraction = 0.5 # Fraction full resolution used for non-rigid registration

# Perform high resolution rigid registration using the MicroRigidRegistrar
start = time.time()
registrar = registration.Valis(slide_src_dir, results_dst_dir, reference_img_f=reference_slide, micro_rigid_registrar_cls=MicroRigidRegistrar)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()

# Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
min_max_size = np.min([np.max(d) for d in img_dims])
img_areas = [np.multiply(*d) for d in img_dims]
max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)

# Perform high resolution non-rigid registration
warper = non_rigid_registrars.SimpleElastixWarper()

micro_reg, micro_error = registrar.register_micro(non_rigid_registrar_cls=warper,
                                                max_non_rigid_registration_dim_px=micro_reg_size,   # or None for full res
                                                align_to_reference=True
                                                )
registrar.warp_and_save_slides(registered_slide_dst_dir, crop="all", non_rigid=True, compression="jpeg", Q=90)


stop = time.time()
elapsed = stop - start
print(f"regisration time is {elapsed/60} minutes")

# We can also plot the high resolution matches using `Valis.draw_matches`:
matches_dst_dir = os.path.join(registrar.dst_dir, "hi_rez_matches")
registrar.draw_matches(matches_dst_dir)

# %%
# Kill the JVM
registration.kill_jvm()
# %%