import os
import re
import cv2
import sys
import json
import pandas as pd
import numpy as np
from PIL import Image
from scipy import ndimage

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from dataloader import (load_img, create_cube, get_working_data,
						get_proto_version, get_calibration_ceramic,
						load_preprocessed, rescale_gt, gt_to_labels)

from img_processing import spectral_correction, correct_defects, preprocess_hsi
#from image_matcher import match_images
from optimizer import optimize_homography
from angle_lib import conv_angles, get_ceramic_dists_normals

from cam_params import cams

from reprojector import sparse_homography, rgb_homography, dist_homography

from utils import check_dirs, overlap_labels

from calibration import adapt_ceramic


def main():
	
	in_path = sys.argv[1]
	json_dict = json.load(open(in_path))

	IDs = json_dict["IDs"]

	raw_hsi_folder = json_dict["raw_hsi_folder"]
	#rgb_hsi_folder = json_dict["rgb_hsi_folder"]

	metadata_file_path = json_dict["metadata_file"]

	adapt_ceramic_folder = json_dict["adapt_ceramic_folder"]

	calibration_files_path = json_dict["calibration_files_path"]
	pprocess_path = json_dict["pre-processed_path"]

	GT_folder = json_dict["GT_folder"]
	raw_cubes_folder = json_dict["raw_cubes_folder"]
	ref_cubes_folder = json_dict["ref_cubes_folder"]
	adj_cubes_folder = json_dict["adj_cubes_folder"]
	perp_cubes_folder = json_dict["perp_cubes_folder"]
	fixed_da_cubes_folder = json_dict["fixed_da_cubes_folder"]
	geom_cubes_folder = json_dict["geom_cubes_folder"]
 
	#desviacion = [0,0.025,0.05,0.1,0.12,0.15]
	desviacion = [0.000001,0.000002,0.000003]
	
	check_dirs(GT_folder, raw_cubes_folder, ref_cubes_folder,
			   adj_cubes_folder, perp_cubes_folder,
			   fixed_da_cubes_folder, geom_cubes_folder)

	metadata_df = pd.read_csv(f"{metadata_file_path}")

	for idp in IDs:

		print(f"---> {idp}")

		nums_list = re.findall("\d+", idp)
		id_num, _ = int(nums_list[0]), int(nums_list[1])

		#rgb_hsi_img = load_img(rgb_hsi_folder, id_num)

		""" cube_dict, gt_dict = load_preprocessed(pprocess_path, idp)
		gt_dict = rescale_gt(cube_dict, gt_dict, idp)
		gt_map = gt_to_labels(gt_dict["groundTruthMap"]) """

		# olap_image = overlap_labels(rgb_hsi_img, gt_map)

		nums_list = re.findall("\d+", idp)
		id_num, _ = int(nums_list[0]), int(nums_list[1])

		proto_v = get_proto_version(metadata_df, idp)
		wd, wa = get_working_data(metadata_df, idp)

		if proto_v<4:
			fixed_wd = .45
		else:
			fixed_wd = .60

		for valor in desviacion:
			hsi_img = load_img(raw_hsi_folder, id_num, valor = None)

			geom_ceramic_cube = load_img(adapt_ceramic_folder, id_num,valor, img_type="npy")

   
			ref_ceramic, dark_img, ref_c_file = get_calibration_ceramic(idp, wd, wa, metadata_df,
																		calibration_files_path,
																		return_dark=True)

			#raw_cube = create_cube(hsi_img)
			raw_cube = hsi_img
			ref_ceramic_cube = create_cube(ref_ceramic)
			dark_cube = create_cube(dark_img)
			

			geom_ceramic_cube = correct_defects(geom_ceramic_cube, geom_ceramic_cube<=5) #Se eliminan valores cercanos a 0
			ref_ceramic_cube = correct_defects(ref_ceramic_cube, ref_ceramic_cube<=5)

			nums_list = re.findall("\d+", ref_c_file)
			c_dist, c_angle = float(nums_list[0])/100, float(nums_list[1])

			ref_dist_map, ref_norm_map = get_ceramic_dists_normals(ref_ceramic_cube, c_dist, c_angle, cams["SN"]["default_K"])
			targ_dist_map, targ_norm_map = get_ceramic_dists_normals(ref_ceramic_cube, wd, wa, cams["SN"]["default_K"])
			perp_ang_dist_map, perp_ang_norm_map = get_ceramic_dists_normals(ref_ceramic_cube, wd, 90, cams["SN"]["default_K"])
			fix_da_dist_map, fix_da_norm_map = get_ceramic_dists_normals(ref_ceramic_cube, fixed_wd, 90, cams["SN"]["default_K"])

			adj_ceramic_cube = adapt_ceramic(np.copy(ref_ceramic_cube), ref_dist_map, ref_norm_map,
											targ_dist_map, targ_norm_map)
			
			perp_ang_ceramic_cube = adapt_ceramic(np.copy(ref_ceramic_cube), ref_dist_map, ref_norm_map,
												perp_ang_dist_map, perp_ang_norm_map)
			
			fix_da_ceramic_cube = adapt_ceramic(np.copy(ref_ceramic_cube), ref_dist_map, ref_norm_map,
												fix_da_dist_map, fix_da_norm_map)


			ref_cube = preprocess_hsi(raw_cube, ref_ceramic_cube, dark_cube)
			adj_cube = preprocess_hsi(raw_cube, adj_ceramic_cube, dark_cube)
			perp_ang_cube = preprocess_hsi(raw_cube, perp_ang_ceramic_cube, dark_cube)
			fix_da_cube = preprocess_hsi(raw_cube, fix_da_ceramic_cube, dark_cube)
			geom_cube = preprocess_hsi(raw_cube, geom_ceramic_cube, dark_cube)
			raw_cube = preprocess_hsi(raw_cube, np.ones_like(raw_cube)*255, dark_cube)

			np.save(f"{ref_cubes_folder}{idp}{valor}.npy", ref_cube)
			np.save(f"{adj_cubes_folder}{idp}{valor}.npy", adj_cube)
			np.save(f"{perp_cubes_folder}{idp}{valor}.npy", perp_ang_cube)
			np.save(f"{fixed_da_cubes_folder}{idp}_{valor}.npy", fix_da_cube)
			np.save(f"{geom_cubes_folder}{idp}_{valor}.npy", geom_cube)	
   			
			
			#np.save(f"{raw_cubes_folder}{idp}_{valor}.npy", raw_cube)

		#np.save(f"{GT_folder}{idp}.npy", gt_map)

		continue

		plt.figure()
		plt.imshow(raw_cube[...,7])

		plt.figure()
		plt.imshow(ref_cube[...,7])

		plt.figure()
		plt.imshow(adj_cube[...,7])

		plt.figure()
		plt.imshow(perp_ang_cube[...,7])

		plt.figure()
		plt.imshow(fix_da_cube[...,7])

		plt.figure()
		plt.imshow(geom_cube[...,7])

		plt.show()
		

		print(wd, wa)
		print(ref_c_file)
		print("\n")
		
		

		# im = Image.fromarray(np.uint8(hsi_img_test))
		# im.save(f"{check_folder}{idp}.png")


if __name__ == "__main__":
	main()