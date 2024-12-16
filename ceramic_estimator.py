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

from ruido import gaussiannoise
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from dataloader import load_img, create_cube, get_working_data, get_proto_version, get_calibration_ceramic

from img_processing import undistort_img, undistort_cube, levels_to_m_lidar, smooth_depth, interpolation_3D_spline
#from image_matcher import match_images
from optimizer import optimize_homography

from angle_lib import conv_angles, get_ceramic_dists_normals, multiscale_angle #,get_hough_normals

from cam_params import cams

from reprojector import sparse_homography, rgb_homography, dist_homography

from utils import check_dirs, print_homography_pts

from calibration import adapt_ceramic


def main():
	
	in_path = sys.argv[1]
	json_dict = json.load(open(in_path))

	IDs = json_dict["IDs"]

	##rgb_folder = json_dict["rgb_folder"]
	dth_folder = json_dict["depth_folder"]
	raw_hsi_folder = json_dict["raw_hsi_folder"]
	##rgb_hsi_folder = json_dict["rgb_hsi_folder"]
	
	##rgb_crop_folder = json_dict["rgb_crop_folder"]
	##dth_crop_folder = json_dict["dth_crop_folder"]

	metadata_file_path = json_dict["metadata_file"]
	##crop_points_file = json_dict["crop_points_file"]
	ceramica_psnr = json_dict["ceramica_psnr"]
	adjusted_t_folder = json_dict["adjusted_t_folder"] #Carpeta con correcciones de translación y rotación

	adapt_ceramic_folder = json_dict["adapt_ceramic_folder"]

	calibration_files_path = json_dict["calibration_files_path"]

	k = json_dict["k"]
	

	#crop_points_dict = json.load(open(crop_points_file))
	
	
	check_folder = "./check_match/"
	
	check_dirs( check_folder,
			   adjusted_t_folder,raw_hsi_folder)

	metadata_df = pd.read_csv(f"{metadata_file_path}")

	pre_errors, post_errors = [], []

	for idp in IDs:

		print(f"---> {idp}")

		nums_list = re.findall("\d+", idp)
		id_num, cap_num = int(nums_list[0]), int(nums_list[1])

		proto_v = get_proto_version(metadata_df, idp)
		wd, wa = get_working_data(metadata_df, idp)

		# rgb_crop = load_img(rgb_crop_folder, id_num)
		# dth_crop = load_img(dth_crop_folder, id_num)
		##rgb_img = load_img(rgb_folder, id_num)
		hsi_img = load_img(raw_hsi_folder, id_num, valor = None)
		##rgb_hsi_img = load_img(rgb_hsi_folder, id_num)

		ref_ceramic, ref_c_file = get_calibration_ceramic(idp, wd, wa, metadata_df,
														  calibration_files_path) #mas cerca al paciente angulo no normalizado, angulo inclinación
		
		#raw_cube = create_cube(hsi_img)
		raw_cube = hsi_img
		ref_ceramic_cube = create_cube(ref_ceramic)

		cube_band = cv2.resize(raw_cube[...,22], cams["SN"]["resolution"][::-1], #cambio resolucion de la camara, se amplia 
						 	   interpolation=cv2.INTER_LINEAR)

		##rgb_hsi_img = cv2.resize(rgb_hsi_img, cams["SN"]["resolution"][::-1],
							  ## interpolation=cv2.INTER_LINEAR)

		raw_cube = cv2.resize(raw_cube, cams["SN"]["resolution"][::-1],
							   interpolation=cv2.INTER_LINEAR)

		ref_ceramic_cube = cv2.resize(ref_ceramic_cube, cams["SN"]["resolution"][::-1],
							   interpolation=cv2.INTER_LINEAR)
		

		if proto_v==4:
			cam_dth = "kinect_dth"
			cam_rgb = "kinect_rgb"
			img_type = "raw"
			interp_type="outliers"
		
		else:
			cam_dth = "l515"
			cam_rgb = "l515"
			img_type = "png"
			interp_type="coords"
			
		
		dth_img = load_img(dth_folder, id_num, valor = None, img_type=img_type)

		##rgb_img = undistort_img(rgb_img, cams[cam_rgb]) #distorsion de la lente, este cams es un diccionario con la camara y sus diccionarios con parametros intrinsecos
		dth_img = undistort_img(dth_img, cams[cam_dth])
		cube_band = undistort_img(cube_band, cams["SN"])
		##rgb_hsi_img = undistort_img(rgb_hsi_img, cams["SN"])
		raw_cube = undistort_cube(raw_cube, cams["SN"])
		ref_ceramic_cube = undistort_cube(ref_ceramic_cube, cams["SN"])

		#crop_pts = [int(pt) for pt in crop_points_dict[idp]]
		##rgb_crop = rgb_img[crop_pts[1]:crop_pts[3], crop_pts[0]:crop_pts[2],:]

		R_ref = np.eye(3)
		t_ref = np.zeros(3)[:,None]
		K_ref = np.copy(cams[cam_dth]["K"])

		R_dest = np.copy(cams[cam_dth]["DTH_to_RGB"]["R"])
		t_dest = np.copy(cams[cam_dth]["DTH_to_RGB"]["t"][:,None])
		K_dest = np.copy(cams[cam_rgb]["K"])

		Rt_ref = np.hstack((R_ref, t_ref))
		Rt_dest = np.hstack((R_dest, t_dest))

		# rgb_img = rgb_homography(rgb_img, dth_img, K_ref, Rt_ref, K_dest, Rt_dest,
		# 								np.copy(cams[cam_dth]["resolution"]), dest_rescaling=1,
		# 								dth_rescaling=2, inpaint_origin=True)
		
		#suvizar mapas de profundidad: lidar coge puntos dispersos e interpola , eliminamos resolucion se quedqa como topografico con profundidad cero.
		if proto_v<4:
			dth_img = levels_to_m_lidar(dth_img, wd)
			dth_img = interpolation_3D_spline(dth_img, cams[cam_dth], f=1/6,
									 		  resamp=.1, b_smooth=5, smooth=.0005)
			# dth_img = cv2.GaussianBlur(dth_img, (5,5), 1)

		else:
			dth_img = (dth_img/1000).astype(np.float32)
			dth_img = cv2.medianBlur(dth_img,5)
			dth_img = smooth_depth(dth_img, r=8, eps=.02)
			# dth_img = cv2.GaussianBlur(dth_img, (9,9), 5)
			# dth_img = interpolation_3D_spline(dth_img, cams[cam_dth], f=1/6,
			# 								  resamp=.1, b_smooth=3, smooth=.0001)
			# dth_img = cv2.bilateralFilter(dth_img.astype(np.float32), 7, 125, 125)


		# invalid_mask = dth_img<0.1
		# _, _, z_angle = conv_angles(dth_img, k, cams["SN"])
		# z_angle = 90-z_angle
		# z_angle[invalid_mask] = 0
		# plt.imsave("./aux_img_.png", z_angle)
		# plt.close()


		adj_t_dict = json.load(open(f"{adjusted_t_folder}{idp}.json"))
		t_mod = np.array(adj_t_dict["t_mod"])[:,None]
		R_mod = np.array(adj_t_dict["R_mod"]).reshape(3,3)
  #vectores de tranalacion entre rgb y snapshot
		

		"""REPROJECT DTH TO SN"""
		
		R_ref = np.eye(3)
		t_ref = np.zeros(3)[:,None]
		K_ref = np.copy(cams[cam_dth]["K"])

		K_dest = np.copy(cams["SN"]["K"])

		# z_angle = get_hough_normals(dth_img, cams["SN"], K=100, T=100)
	#############################################################################
		# z_angle = multiscale_angle(sn_dth, 3, cams["SN"])
		# z_angle = (90.-z_angle).astype(np.float32)

		Rt_ref = np.hstack((R_ref, t_ref))
		Rt_dest_mod = np.hstack((R_mod, t_mod))
  
		#desviacion = [0,0.05,0.1,0.12,0.15]
		desviacion = [0.000001,0.000002,0.000003]
		rmse = []
		psnr = []
  
		for valor in desviacion:
			
			Rt_dest_mod,rmse_value,psnr_value = gaussiannoise(Rt_dest_mod,valor)
			
			#print("u rango:", Rt_dest_mod.min(), Rt_dest_mod.max())
			
			#print(Rt_dest_mod)
			#mueve la profundidad a sanpshot/ dest parametros recalculados de la snapshot
			sn_dth = dist_homography(dth_img, K_ref, Rt_ref, K_dest, Rt_dest_mod,
									np.copy(cams["SN"]["resolution"]), dth_rescaling=1,
									dest_rescaling=1/5, interp_result=interp_type, keep_rescaling=True)
   
			while np.isnan(sn_dth).sum() > 0:
				Rt_dest_mod,rmse_value,psnr_value = gaussiannoise(Rt_dest_mod,valor)
    
				sn_dth = dist_homography(dth_img, K_ref, Rt_ref, K_dest, Rt_dest_mod,
									np.copy(cams["SN"]["resolution"]), dth_rescaling=1,
									dest_rescaling=1/5, interp_result=interp_type, keep_rescaling=True)
    
			rmse.append(rmse_value)
			psnr.append(psnr_value)
			
			# plt.imsave(f"./sn_dth_{idp}.png", sn_dth)
			# Image.fromarray(np.uint8(100*sn_dth)).save(f"./sn_dth_{idp}.png")
			sn_dth = smooth_depth(sn_dth, r=6, eps=.02)
			
			z_angle = multiscale_angle(sn_dth, 3, cams["SN"])
			z_angle = (90.-z_angle).astype(np.float32)
			

			#z_angle = get_hough_normals(sn_dth, cams["SN"], K=100, T=100)

			""" rgb_hsi_img = cv2.resize(rgb_hsi_img, cams["SN_std"]["resolution"][::-1],
								interpolation=cv2.INTER_LINEAR)
			edges = cv2.Canny(cv2.GaussianBlur(rgb_hsi_img, (3,3),0), 1000, 2000, apertureSize=5)

			z_angle_edge = np.copy(z_angle)
			z_angle_edge[edges>0] = 0
			# rgb_hsi_img[edges>0,:] = [0,0,0]
			
			
			# plt.imsave("./edges_img.png", z_angle)
			# plt.imsave("./rgb_edges_img.png", rgb_hsi_img)
			plt.imsave(f"./norms_check/{idp}.png", z_angle_edge, vmin=0, vmax=90)
			np.save(f"./angle_maps/{idp}.npy", z_angle) """
			# quit()
			

			# np.save(f"./norms_img/{idp}.npy", z_angle)

			nums_list = re.findall("\d+", ref_c_file) #lee la distancia y angulo
			c_dist, c_angle = float(nums_list[0])/100, float(nums_list[1])
			ref_dist_map, ref_norm_map = get_ceramic_dists_normals(z_angle, c_dist, c_angle, cams["SN_std"]["K"]) #compensar angulo y distancia calculadado y real

			ref_ceramic_cube = cv2.resize(ref_ceramic_cube.astype(float), cams["SN"]["default_resolution"][::-1],
									interpolation=cv2.INTER_LINEAR)
			
			sn_ceramic = adapt_ceramic(ref_ceramic_cube, ref_dist_map,
									ref_norm_map, sn_dth, z_angle)
			#print(sn_ceramic)
			print("devuelb¡ve nana:",np.isnan(sn_ceramic).sum())
			np.save(f"{adapt_ceramic_folder}{idp}_{valor}.npy", sn_ceramic)
			print("ceramica guardada",valor)

			np.save(f"{adapt_ceramic_folder}{idp}_{valor}.npy", sn_ceramic)

		"""plt.figure(figsize=(10, 5))

		 # Gráfico de RMSE
		plt.subplot(1, 2, 1)
		plt.plot(desviacion, rmse, marker='o', label='RMSE', color='red')
		plt.title('RMSE por iteración')
		plt.xlabel('Iteración')
		plt.ylabel('RMSE')
		plt.grid(True)
		plt.legend()

		# Gráfico de PSNR
		plt.subplot(1, 2, 2)
		plt.plot(desviacion, psnr, marker='o', label='PSNR', color='blue')
		plt.title('PSNR por iteración')
		plt.xlabel('Iteración')
		plt.ylabel('PSNR (dB)')
		plt.grid(True)
		plt.legend()

		plt.tight_layout()
		plt.savefig(f"{ceramica_psnr}_{idp}_rmse_psnr_plot.png") 
		plt.show() """
  
		
		continue
		
		

		cal_cube = raw_cube/(sn_ceramic+1e-6)

		cal_cube[(sn_ceramic==0) * (sn_ceramic>255)] = 0

		sn_ceramic[sn_ceramic>255] = 0

		cal_cube[cal_cube>2] = 0


		plt.figure()
		plt.imshow(raw_cube[...,7])

		plt.figure()
		plt.imshow(cal_cube[...,7])

		plt.figure()
		plt.imshow(sn_ceramic[...,7])

		plt.figure()
		plt.imshow(sn_dth)

		plt.figure()
		plt.imshow(np.uint8(z_angle))
		plt.show()


		# im = Image.fromarray(np.uint8(hsi_img_test))
		# im.save(f"{check_folder}{idp}.png")


if __name__ == "__main__":
	main()