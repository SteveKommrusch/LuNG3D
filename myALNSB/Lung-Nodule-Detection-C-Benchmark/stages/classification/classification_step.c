/**
 * classification_step.c: this file is part of the ALNSB project.
 *
 * ALNSB: the Adaptive Lung Nodule Screening Benchmark
 *
 * Copyright (C) 2014,2015 University of California Los Angeles
 *
 * This program can be redistributed and/or modified under the terms
 * of the license specified in the LICENSE.txt file at the root of the
 * project.
 *
 * Contact: Alex Bui <buia@mii.ucla.edu>
 *
 * Experimental edits done by Steve Kommrusch
 *
 */
/**
 * Written by: Shiwen Shen, Prashant Rawat, Louis-Noel Pouchet and William Hsu
 *
 */
#include <math.h>
#include <stages/classification/classification_step.h>
#include <utilities/file_io.h>
#include <toolbox/bwconncomp.h>
#include <stdlib.h>


static
float* compute_mean_mat (float* data, int sz_dim_1, int sz_dim_2, int dim_id)
{
  float* res = NULL;
  int res_sz = 0;
  if (dim_id == 1)
    {
      res = (float*) malloc (sizeof(float) * sz_dim_2);
      res_sz = sz_dim_2;
    }
  else if (dim_id == 2)
    {
      res = (float*) malloc (sizeof(float) * sz_dim_1);
      res_sz = sz_dim_1;
    }
  else
    {
      fprintf (stderr, "[ERROR][classification] Unsupported dimension id\n");
      exit (1);
    }
  float (*im2D)[sz_dim_2] = (float (*)[sz_dim_2])(data);
  int i, j;

  for (i = 0; i < res_sz; ++i)
    res[i] = 0;

  for (i = 0; i < sz_dim_1; ++i)
    for (j = 0; j < sz_dim_2; ++j)
      {
	int pos_res = dim_id == 1 ? j : i;
	res[pos_res] += im2D[i][j];
      }
  if (dim_id == 1)
    for (i = 0; i < res_sz; ++i)
      res[i] /= (float)sz_dim_1;
  else
    for (i = 0; i < res_sz; ++i)
      res[i] /= (float)sz_dim_2;

  /* printf ("sz_dim_1=%d, res_sz=%d, sz_dim_2=%d\n", sz_dim_1, res_sz, sz_dim_2); */

  /* for (i = 0; i < sz_dim_1; ++i) */
  /*   debug_print_featureVector ("feat", data + i * sz_dim_2, sz_dim_2); */


  return res;
}

static
void debug_print_featureVector (char* msg, float* v, int nb_feat)
{
  int i;
  printf ("%s:\n", msg);
  for (i = 0; i < nb_feat; ++i)
    printf ("%.2f ", v[i]);
  printf ("\n");
}


void classification_cpu (s_alnsb_environment_t* __ALNSB_RESTRICT_PTR env,
			 image3DReal* __ALNSB_RESTRICT_PTR inputPrep,
			 image3DBin* __ALNSB_RESTRICT_PTR inputPresel,
			 image3DReal* __ALNSB_RESTRICT_PTR inputFeats,
			 image3DReal** __ALNSB_RESTRICT_PTR output)
{
  // Allocate output data.
  *output = image3DReal_alloc (inputPrep->slices, inputPrep->rows, inputPrep->cols);
  // 1D view of output data.
  ALNSB_IMRealTo1D(*output, out_img);
  // 1D view of input data.
  ALNSB_IMBinTo1D(inputPresel, in_img);
  ALNSB_IMRealTo1D(inputPrep, base_img);
  ALNSB_IMRealTo1D(inputFeats, feats);

  int debug = env->verbose_level;

  // Classifier info.
  int number_of_features = env->classifier_num_features;
  size_t number_of_pos_samples;
  size_t number_of_neg_samples;

  int xc = inputPresel->rows;
  int yc = inputPresel->cols;
  int zc = inputPresel->slices;
  unsigned int sz = xc * yc * zc;

  unsigned int i, j;
  int** comp_coordinates = NULL;
  int* comp_sz = NULL;
  int num_candidate_nodules = 0;

  FILE* fp;
  int overridenum=0;
  int overridekeep;

  fp = fopen("segInputOverride.csv","r");
  if (fp) {
    /* overnum is a flag and counter.  If set, accept first 10 */
    /* samples, then probabilistically accept elements to target averages */
    overridenum = 1;
    fclose(fp);
  }

  alnsb_bwconncomp_bin (in_img, zc, xc, yc,
			&comp_coordinates, &comp_sz, &num_candidate_nodules,
			NULL, 1);

  int featureMask[number_of_features];
  for (i = 0; i < number_of_features; ++i)
    featureMask[i] = env->classifier_active_features[i];
  if (debug == 42)
    {
      printf ("feature mask:\n");
      for (i = 0; i < number_of_features; ++i)
  	printf ("f%d=%d\n", i+1, featureMask[i]);
      printf ("\n");
    }

  float xyzSpace[] = { env->scanner_pixel_spacing_x_mm,
		       env->scanner_pixel_spacing_y_mm,
		       env->scanner_slice_thickness_mm };
  /// FIXME: ugly. Should be a global define. Sets an upper bound on
  /// number of samples in any possible classifier matrix.
  size_t max_nb_entries = 10000 * number_of_features;
  float* selectedNegativeSamples =
    alnsb_read_data_from_binary_file_nosz
    (env->classifier_negative_featMat_filename, sizeof(float),
     max_nb_entries, &number_of_neg_samples);
  number_of_neg_samples /= number_of_features;
  float* selectedPositiveSamples =
    alnsb_read_data_from_binary_file_nosz
    (env->classifier_positive_featMat_filename, sizeof(float),
     max_nb_entries, &number_of_pos_samples);
  number_of_pos_samples /= number_of_features;
  float* meanFeature =
    alnsb_read_data_from_binary_file
    (env->classifier_meanFeat_filename, sizeof(float), number_of_features);
  float* stdFeature =
    alnsb_read_data_from_binary_file
    (env->classifier_stdFeat_filename, sizeof(float), number_of_features);

  if (debug > 2)
    printf ("posSamplesMat: %d samples, negSamplesMat: %d samples\n",
	    number_of_pos_samples, number_of_neg_samples);
  
  float* meanP = compute_mean_mat (selectedPositiveSamples,
				   number_of_pos_samples,
				   number_of_features, 1);
  float* meanN = compute_mean_mat (selectedNegativeSamples,
				   number_of_neg_samples,
				   number_of_features, 1);

  if (debug > 2)
    {
      debug_print_featureVector ("meanP", meanP, number_of_features);
      debug_print_featureVector ("meanN", meanN, number_of_features);
      debug_print_featureVector ("meanFeat", meanFeature, number_of_features);
      debug_print_featureVector ("stdFeat", stdFeature, number_of_features);
    }


  for (i = 0; i < number_of_features; ++i)
    {
      if (featureMask[i] == 0)
	{
	  meanP[i] = 0;
	  meanN[i] = 0;
	  meanFeature[i] = 0;
	  stdFeature[i] = 0;
	}
    }

  int noduleNum = 0;
  float featVect[number_of_features];
  /* 12 vectors in LuNG analysis, initialize averages to match 51 nodules */
  float tgtVect[12] = {17.390,7.604,18.497, 0.591, 80.886,1.506,
                        0.651,0.498, 5.633,12.121,  2.608,0.436};
  float stdVect[12] = {33.973,5.670,15.062, 0.283,202.364,0.209,
                        0.216,0.219, 4.009, 9.783,  3.248,0.187};
  float keepCalc;
  float avgVect[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
  unsigned int offset = 0;
  printf ("SJKINFO: xyzSpace[0]=%.3f, xyzSpace[1]=%.3f, xyzSpace[2]=%.3f\n",xyzSpace[0], xyzSpace[1], xyzSpace[2]);
  for (i = 0; i < num_candidate_nodules; ++i)
    {
      if (debug > 1)
	printf ("classify nodule #%d\n", i);
      for (j = 0; j < number_of_features; ++j)
	if (featureMask[j] != 0)
	  featVect[j] = feats[j + offset];
	else
	  featVect[j] = 0;
      if (overridenum >0) {
        overridekeep = 1;
        for (j = 0; j < 12; ++j) {
          if (overridenum <= 20) {
            avgVect[j] += 0.05*featVect[j];
          } else {
            if (featVect[j] < tgtVect[j] && avgVect[j] < tgtVect[j]) {
              keepCalc = (4*tgtVect[j] - featVect[j] - 3*avgVect[j])/stdVect[j];
              printf("SJK: Low %d: f=%.2f, a=%.2f, t=%.2f, k=%.2f, r=%.2f\n",
                     j,featVect[j],avgVect[j],tgtVect[j],keepCalc,(float)(rand()%512)/511.0);
              if ((keepCalc > 3.0) && ((float)(rand()%512)/511.0 > (0.7 + 0.9/keepCalc)))
                overridekeep=0;
            } else if (featVect[j] > tgtVect[j] && avgVect[j] > tgtVect[j]) {
              keepCalc = (featVect[j] + 3*avgVect[j] - 4*tgtVect[j])/stdVect[j];
              printf("SJK: High %d: f=%.2f, a=%.2f, t=%.2f, k=%.2f, r=%.2f\n",
                     j,featVect[j],avgVect[j],tgtVect[j],keepCalc,(float)(rand()%512)/511.0);
              if ((keepCalc > 3.0) && ((float)(rand()%512)/511.0 > (0.7 + 0.9/keepCalc)))
                overridekeep=0;
            }
          }
        }
        if (overridekeep) {
          printf("KEEPFEAT: \n");
          for (j = 0; j < 12; ++j) {
            /* Update averages with 0.95*avg + 0.05*feat */
            avgVect[j] = 0.95*avgVect[j] + 0.05*featVect[j];
            printf ("%.2f ", featVect[j]);
          }
          printf("\n");
          overridenum++;
        }
      }
      if (debug > 2)
	debug_print_featureVector ("featVec", featVect, number_of_features);
      for (j = 0; j < number_of_features; ++j)
	if (featureMask[j] != 0)
	  featVect[j] = (featVect[j] - meanFeature[j]) / stdFeature[j];
      if (debug > 2)
	debug_print_featureVector ("featVecNorm", featVect, number_of_features);
      float temp1 = 0;
      float temp2 = 0;
      for (j = 0; j < number_of_features; ++j)
	if (featureMask[j] != 0)
	  {
	    temp1 += pow(featVect[j] - meanP[j], 2);
	    temp2 += pow(featVect[j] - meanN[j], 2);
	  }
      if (debug > 2){
        // debug_print_featureVector ("meanP_after_mask", meanP, number_of_features);
        // debug_print_featureVector ("meanN_after_mask", meanN, number_of_features);
        printf ("distance to pos: %f, distance to neg: %f\n", temp1, temp2);
      }
  
      int z_plane =
	comp_coordinates[i][0] / (xc * yc);

      int start_z= comp_coordinates[i][0] / (xc * yc);
      int end_z= comp_coordinates[i][comp_sz[i]-1] / (xc * yc);
      int start_x = xc-1;
      int end_x = 0;
      int start_y = yc-1;
      int end_y = 0;
      int base_x;
      int base_y;
      int base_z;
      int xy;
      for (j = 0; j < comp_sz[i]; ++j) {
        xy = comp_coordinates[i][j] % (xc*yc);
        if (start_y > xy/xc)
          start_y = xy/xc;
        if (end_y < xy/xc)
          end_y = xy/xc;
        if (start_x > xy%xc)
          start_x = xy%xc;
        if (end_x < xy%xc)
          end_x = xy%xc;
      }
      // Create images limited to 40x40x20 in size
      printf ("SJKINFO: Nodule %d start,end_x = %d,%d, start,end_y = %d,%d, start,end_z = %d,%d\n",i,start_x,end_x,start_y,end_y,start_z,end_z);
      end_x -= (end_x - start_x - 39)/2;
      start_x += end_x - start_x - 39;
      end_y -= (end_y - start_y - 39)/2;
      start_y += end_y - start_y - 39;
      end_z -= (end_z - start_z - 19)/2;
      start_z += end_z - start_z - 19;
      printf ("SJKINFO: 40x40x20: start,end_x = %d,%d, start,end_y = %d,%d, start,end_z = %d,%d\n",start_x,end_x,start_y,end_y,start_z,end_z);
      
      printf ("SJKINFO: Nodule %d comp_sz %d, feature[4] = %f, xc = %d, yc = %d, zc = %d\n",i,comp_sz[i],featVect[4],xc,yc,zc);
      for (j = 0; j < comp_sz[i]; ++j) {
        base_z = comp_coordinates[i][j]/(xc*yc);
        base_y = (comp_coordinates[i][j]%(xc*yc))/xc;
        base_x = (comp_coordinates[i][j]%(xc*yc))%xc;
        if ((base_x < start_x) || 
            (base_x > end_x) || 
            (base_y < start_y) || 
            (base_y > end_y) || 
            (base_z < start_z) || 
            (base_z > end_z)) {
          printf ("SJK: Clipping out pixel for %d at x=%d,y=%d,z=%d\n",i,base_x,base_y,base_z);
          continue;  // Ignore pixels that are too far out of normal
        }
        if (temp1 < temp2) {
          // Image is suspicious - put on slice 0
          out_img[i*xc*40 + (base_z - start_z)*40 + (base_x - start_x) + (base_y - start_y)*xc] 
               = base_img[comp_coordinates[i][j]];
        } 
        // Put all images starting on slice 4
        out_img[4*xc*yc+i*xc*40 + (base_z - start_z)*40 + (base_x - start_x) + (base_y - start_y)*xc] 
             = base_img[comp_coordinates[i][j]];
      }
      // Print volume data for machine learning as CSV lines
      // Note that feature data for pre-training is printed out above
      if (overridenum > 0 && overridekeep) {
        printf("KEEPCSV: ");
      } else {
        printf("IMAGECSV: ");
      }
      for (base_z = start_z; base_z <= end_z; base_z++) {
        for (base_y = start_y; base_y <= end_y; base_y++) {
          for (base_x = start_x; base_x <= end_x; base_x++) {
            if ((base_x < 0) || (base_x >= xc) ||(base_y < 0) || (base_y >= yc) ||(base_z < 0) || (base_z >= zc)) {
              printf ("0.000,");
            } else {
              printf ("%.3f,", out_img[4*xc*yc+i*xc*40 + (base_z - start_z)*40 + (base_x - start_x) + (base_y - start_y)*xc]);
            }
          }
        }
      }
      printf("\n");
      if (temp1 < temp2) {
        // Image is suspicious
        ++noduleNum;
        float nodule_volume = comp_sz[i] * xyzSpace[0] * xyzSpace[1] * xyzSpace[2];
        if (debug)
          printf ("[INFO] Suspicous nodule #%d: slice=%d volume=%.2f\n", noduleNum, z_plane, nodule_volume);
      }
      offset += number_of_features;
    }
  for (i = 0; i < num_candidate_nodules; ++i)
    free (comp_coordinates[i]);
  free (comp_coordinates);
  free (comp_sz);

  printf ("[INFO] Retained %d nodules out of %d candidates\n", noduleNum, num_candidate_nodules);
}
