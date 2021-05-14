# **Soil Properties Estimation Using Deep Regression Neural Network**
## **NTB-TB data points**
![image](figures/ntb-tb.png)
## **Sample test area using trained model from NTB-TB data**

![image](figures/test_area_official.png)


`X_STEP = 3`

`Y_STEP = 3`

`ELEVATION_STEP = 2`

### **Soil type assumption**

Soil type number. From down to bottom.

`assign_num_list = {`
                    `'topsoil layer': 7,`
                    `'reclaimed layer': 6,`
                    `'sedimentary layer': 5,`
                    `'weathered soil': 4,`
                    `'weathered rock': 3,`
                    `'soft rock': 2,`
                    `'moderate rock': 1,`
                     `'hard rock': 0,`
                  `}`
                 
 ### **Type color assumption**
 
`label_colours = ['black', 'brown', 'red', 'magenta',`
                `'pink', 'green',`
                `'blue','cyan']`
                
## **Predicted results - 3D point grid**

![image](figures/3d_ntb_tb.png)

![image](figures/3d_points_grid_train_on_merged_NTB_TB.png)

**With these 3D Data points, we can extract the specific cross section. Sample as follow**

### **X-X Cross Section**
![image](figures/cross_section_X_ntb_tb.png)

### **Y-Y Cross Section**
![image](figures/cross_section_Y_ntb_tb.png)


# Tutorial
- Step 1: Using [Step1_develop_regression_models.ipynb](Step1_develop_regression_models.ipynb) to training model base on NTB or TB or both.
- Step 2: Using [Step2_generating_estimated_results_visualize_3_soil_types.ipynb](Step2_generating_estimated_results_visualize_3_soil_types.ipynb) to generate test area predicted results
- Step 3: Using [Step3_visualize_3D_points_and_cross_sections_visualize_3_soil_types.ipynb](Step3_visualize_3D_points_and_cross_sections_visualize_3_soil_types.ipynb) for visualizing cross section interpolation images.

# Notes
- Check **model architecture** in [Step1_develop_regression_models.ipynb](Step1_develop_regression_models.ipynb) before making prediction.
