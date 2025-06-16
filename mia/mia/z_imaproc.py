import sys
import os
from scipy.stats import skew, kurtosis
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture
import math
from scipy.spatial import distance_matrix

from z_stat import * # calculate_morans_i, calculate_variogram, calculate_standard_deviation, calculate_general_stat

from tqdm import tqdm
import cProfile

print("Process Started. Please wait till end...")

upperbound=255
lowerbound=5
RC_thresh=0.1
Dep_thresh=80
sv1=110
sv2=0.273 #80
sv3=0
sv4=sv5=0.5
color_spec="HIP Perceived Grayscale Analysis"
max_std_dev = max_moran_I=max_dnsi=0
Dep_thresh_flag=0
max_varigram = 1
flg_hough=0

LL_vario = 1.0          # Maximum value of the sigmoid function
KK_vario = 10.0         # Steepness of the sigmoid curve
x0_vario = 0.8 

LL_dnsi = 1.0          # Maximum value of the sigmoid function
KK_dnsi = 10.0         # Steepness of the sigmoid curve
x0_dnsi = 0.8 






from scipy.ndimage import generic_filter

def homogeneous_masking(image, output_path, flg, neighborhood_size=3, background_color='black'):
    global lowerbound, Dep_thresh, RC_thresh, Dep_thresh_flag, max_varigram, L, K, x0

    # Ensure the image is loaded properly
    if image is None or image.size == 0:
        raise ValueError("Error: Image is empty or not loaded correctly.")

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure uint8 type
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    height, width = image.shape
    half_size = neighborhood_size // 2

    # Initialize new image for filtering
    new_img = image.copy()

    # Apply a local standard deviation filter
    def local_std(patch):
        return np.std(patch)

    std_dev_map = generic_filter(image, local_std, size=(neighborhood_size, neighborhood_size))

    # Suppress pixels with low standard deviation
    new_img[std_dev_map <= lowerbound] = 0 

    # Show processed image
    plt.imshow(new_img, cmap='gray')
    plt.axis('off')
    plt.show()

    gray = new_img
    '''
    def hough_transform_remove_circles(gray):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Dynamically determine circle radius limits
        minRadius = max(5, int(min(height, width) * 0.02))  
        maxRadius = max(6, int(min(height, width) * 0.1))  

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20, 
            param1=50, 
            param2=30, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        # Create a copy of the original image
        result = gray.copy()

        # If circles are detected, replace them with black color
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                # Replace the circle region with black (0 intensity)
                cv2.circle(result, center, radius, 0, -1)

        else:
            print("Warning: No circles detected.")
        return result


    '''
    
    def hough_transform_remove_circles(gray):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Get image dimensions
        height, width = gray.shape[:2]

        # Dynamically determine circle radius limits
        minRadius = max(5, int(min(height, width) * 0.02))  
        maxRadius = max(6, int(min(height, width) * 0.1))  

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20, 
            param1=50, 
            param2=30, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        # Create a copy of the original image
        result = gray.copy()

        # If circles are detected, process them
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, radius = circle  # Center coordinates and radius

                # Define a circular mask for the central region (inner 50% of radius)
                mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), max(1, radius // 2), 255, -1)  # Create circular mask
                
                # Extract pixel values inside the mask
                circle_pixels = gray[mask == 255]

                # Compute the average intensity of the central region
                avg_intensity = np.mean(circle_pixels) if circle_pixels.size > 0 else 0

                # Remove circle only if the average intensity > 200
                print("avg int****", avg_intensity)
                if avg_intensity > a:
                    cv2.circle(result, (cx, cy), radius, 0, -1)  # Fill circle with black

        else:
            print("Warning: No circles detected.")

        return result
    
    
    if(flg==1):
        result = hough_transform_remove_circles(gray)
    else:
        result=new_img
    # Save the modified image
    cv2.imwrite(output_path, result)
    print(f"Processed image saved to {output_path}")

    # Display the output
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.show()
    return result





    




def gap_tunning(original_image):
    canny = cv2.Canny(original_image, threshold1=100, threshold2=200)
    canny_copy = canny.copy()
    #output_image = np.ones_like(original_image) * 255
    #output_image[edges_copy == 255] = 255
    original_image[canny_copy == 255] = 255

def remaining_min(left_seg):
    return min(left_seg, 255-left_seg)
    




import numpy as np

def adjacency_shift(values, flag):
    # Ensure values is a NumPy array
    values = np.array(values, dtype=np.float64)
    
    # Check if it's 1D and reshape it to 2D (e.g., (1, n) or (n, 1) depending on your patch)
    if values.ndim == 1:
        values = values.reshape(1, -1)  # Reshapes the 1D array to 2D (1, n) array
        #print(f"Input reshaped to 2D: {values.shape}")
    
    # Ensure it's a 2D array
    if values.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")
    
    rows, cols = values.shape
    neighbors_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Horizontal & Vertical
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
    numerator = 0
    S0 = 0
    
    # Iterate over all pixels
    for i in range(rows):
        for j in range(cols):
            for offset in neighbors_offsets:
                ni, nj = i + offset[0], j + offset[1]

                # Check if neighbor is within bounds
                if 0 <= ni < rows and 0 <= nj < cols:
                    numerator = numerator + (values[i, j] - values[ni, nj]) ** 2
                    S0 += 1
    
    valley_well = numerator / 2
    return valley_well



# Degree of centrality function
def calculate_degree_centrality(patch):
    n = patch.shape[0]  # n x n patch
    central_pixel = patch[n//2, n//2]  # Central pixel intensity value
    
    # Define directions (including diagonals) and corresponding offset positions
    directions = {
        "up": patch[:n//2, n//2],              # Top half
        "down": patch[n//2+1:, n//2],         # Bottom half
        "left": patch[n//2, :n//2],           # Left half
        "right": patch[n//2, n//2+1:],        # Right half
        "top-left": patch[:n//2, :n//2],      # Top-left diagonal
        "top-right": patch[:n//2, n//2+1:],   # Top-right diagonal
        "bottom-left": patch[n//2+1:, :n//2], # Bottom-left diagonal
        "bottom-right": patch[n//2+1:, n//2+1:] # Bottom-right diagonal
    }
    
    # Thresholds
    intensity_threshold = 50
    adjacency_threshold = 500
    
    dissimilar_pixels = 0  # Counter for dissimilar pixels
    
    for direction, values in directions.items():
        # Calculate the median intensity in each direction
        median_intensity = np.median(values)
        
        # Calculate adjacency shift for the current direction (using 2D array of intensity values)
        adj_shift_value = adjacency_shift(values, flag=0)  # flag=0 for normal operation
        
        # Check for intensity difference
        intensity_diff = abs(central_pixel - median_intensity)
        
        # Check for adjacency shift threshold
        if intensity_diff > intensity_threshold or adj_shift_value > adjacency_threshold:
            dissimilar_pixels += 1
    
    # Degree of centrality (higher is more central)
    degree_centrality = (8 - dissimilar_pixels)  # 8 directions (4 main directions + 4 diagonals)
    
    return degree_centrality































def read_sample(file_path, image, neighborhood_size):
    global upperbound, lowerbound, RC_thresh, Dep_thresh, color_spec, Dep_thresh_flag
  
    # Load the Excel file
    file_path = file_path  # Replace with your actual file path
    df = pd.read_excel(file_path)
    #print(df)
    # Ensure the columns 'R', 'G', 'B' exist in the DataFrame
    for col in ['GS1S','GS2S','GS1RC','GS2RC','GS1Dep','GS2Dep']:
        # Calculate mean, standard deviation, and skewness
        mean_std = df[['GS1S','GS2S','GS1RC','GS2RC','GS1Dep','GS2Dep']].describe()
        print("Other Statistical Data:" "\n========================\n", mean_std)
        



        
        RC_thresh=round(mean_std.loc["min"]["GS2RC"],2)+(round(max_moran_I,2)-round(mean_std.loc["min"]["GS2RC"],2))*sv4
        if (Dep_thresh_flag==1):
            Dep_thresh_controll=10*math.log(round(max_varigram,2)-round(mean_std.loc["min"]["GS2Dep"],2))*sv5
            Dep_thresh=round(mean_std.loc["min"]["GS2Dep"],2)+ round(3.5 * round(mean_std.loc["std"]["GS1S"],2),2) +   Dep_thresh_controll     
        else:  
            Dep_thresh=round(max_varigram/2) *sv5 
            print("////////////",Dep_thresh)
        
         
        # Calculate the threshold for white pixels
        def calculate_upperbound():
                global upperbound, lowerbound
                upperbound=20

        # Calculate the threshold for black pixels
        def calculate_lowerbound():
            global upperbound, lowerbound 
            if color_spec=="HIP Perceived Grayscale Analysis":
                lowerbound=round(mean_std.loc["mean"]["GS1S"],2) + round(3.5 * round(mean_std.loc["std"]["GS1S"],2),2)
                
            else:
                lowerbound=round(mean_std.loc["mean"]["GS2S"],2) + round(3.5 * round(mean_std.loc["std"]["GS2S"],2),2) 
                
        
        
        calculate_upperbound()     
        calculate_lowerbound()
        
        lowerbound=lowerbound+lowerbound*sv3*2
        
        
        print("*************","\nRelational Threshold Set:", round(RC_thresh,2), "\nDependancy Threshold Set:", round(Dep_thresh,2)) 
        print("*************","\nlowerbound:", lowerbound, "\nupperbound:", upperbound) 
        print("\n*****************\nsv3:",sv3,"sv4:",sv4,"sv5:",sv5)
        
        return mean_std






def display_matplt(image, new_img):
    # Plot original and classified image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title("new_img (Texture vs. Background)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("xxx.png", bbox_inches='tight', dpi=300)
    plt.show()






def Eor_dia(binary_mask,output_file):
    '''
    mask = cv2.imread(output_file, cv2.IMREAD_GRAYSCALE)


    # Ensure binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    '''
    
    # Step 1: Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Create a filled mask
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)



    # Compute original area
    original_area = np.sum(binary_mask > 0)
    target_area = original_area * 0.40  # Retain 90% (erode 10%)

    # Define a structuring element (kernel)
    kernel = np.ones((3, 3), np.uint8)  # Small kernel for shape preservation

    # Iteratively erode while maintaining shape
    eroded_mask = filled_mask.copy()

    while True:
        temp_mask = cv2.erode(eroded_mask, kernel, iterations=1)
        new_area = np.sum(temp_mask > 0)

        # Stop when the remaining area reaches the target
        if new_area <= target_area:
            break  

        eroded_mask = temp_mask  # Update the eroded object

    # Save or visualize the result
    cv2.imwrite('Final_'+output_file, eroded_mask)
    cv2.imshow('Eroded Object (10%)', eroded_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







countcount = 0
def fl_math(intensity_value,mean, median, modd, sv1,sv2):
    global countcount
    
    input_px=intensity_value
    #if(dc>3):
    input_px = max(intensity_value, mean, median, modd)

    a = int(sv1) - int(sv2 * remaining_min(sv1))  # DEFAULT 80
    #print("a is=======================", a)
    alpha = int(sv1)  # c in case of u_d, DEFAULT 110
    b = alpha  # DEFAULT 110
    beta = alpha  # a in case of u_b, DEFAULT 110
    c = int(sv1) + int(sv2 * remaining_min(sv1))  # DEFAULT 140

    if countcount == 0:
        print("a:", a, "alpha:", alpha, "b:", b, "beta:", beta, "c:", c)
        countcount += 1
    
    def u_d(input_px):
        if input_px <= a:
            output = 1
        else:
            output = max((alpha - input_px) / (alpha - a), 0)
        return output

    def u_g(input_px):
        output = max(min((input_px - a) / (b - a), (c - input_px) / (c - b)), 0)
        return output

    def u_b(input_px):
        if input_px >= c:
            output = 1
        else:
            output = max((input_px - beta) / (c - beta), 0)
        return output

    def process_pixel(intensity_value):
        v_d = 0
        v_g = 127
        v_b = 255

        num = v_d * u_d(input_px) + v_g * u_g(input_px) + v_b * u_b(input_px)
        den = u_d(input_px) + u_g(input_px) + u_b(input_px)
        output = int(num / den)

        return int(output)

    # Compute degree centrality for the current pixel (row, col)
    #degree_centrality = calculate_degree_centrality(row, col, patch)
    #print(f"Degree Centrality of pixel ({row},{col}): {degree_centrality}")

    # Adjust the input based on degree centrality
    #adjusted_input = input_px * (1 + degree_centrality * 0.05)  # Adjust by degree centrality
    
    x = process_pixel(int(input_px))
    return x








    






def vote(intensity_value,fl_pix_val,vario_std,R_dnsi_by_std,G_dnsi_by_std,B_dnsi_by_std,RR,GG,BB):
    

        if((GG<100 and GG<RR and GG<BB) and (R_dnsi_by_std>G_dnsi_by_std and G_dnsi_by_std>B_dnsi_by_std)):
            return 0
        else:
            return 255

    





 



def extract_and_classify_image_gray(img_colr,image, output_file, neighborhood_size, sv1, sv2):
    """
    Extract spatial statistics (Moran's I, CSSNI, Variogram, Standard Deviation)
    for each pixel in the image and classify texture regions using GMM clustering.
    """
    ccc=0
    a = int(sv1) - int(sv2 * remaining_min(sv1))  # DEFAULT 80
    
    
    img_colr_R = img_colr[:,:,0].astype(float)
    img_colr_G = img_colr[:,:,1].astype(float)
    img_colr_B = img_colr[:,:,2].astype(float)
    
    
         
    image=homogeneous_masking(image, "../output/xxx_Houg.jpg", flg_hough, neighborhood_size, background_color='black')
    zero_count = np.sum(image == 0)
    print("zero_count+++++++++++++++++++++++", zero_count)
    
    new_img=image.copy()
    y_coords, x_coords = np.where(image > 0)
    height, width = image.shape

    patch_radius = neighborhood_size // 2  # e.g., neighborhood_size=5 → radius=2
    padded_image = np.pad(image, pad_width=patch_radius, mode='reflect')
    R_padded_image = np.pad(img_colr_R, pad_width=patch_radius, mode='reflect')
    G_padded_image = np.pad(img_colr_G, pad_width=patch_radius, mode='reflect')    
    B_padded_image = np.pad(img_colr_B, pad_width=patch_radius, mode='reflect')
  
    
    for x, y in zip(x_coords, y_coords):
            if new_img[y,x]>a:
                patch = padded_image[y:y+2*patch_radius+1, x:x+2*patch_radius+1]
                R_patch = R_padded_image[y:y+2*patch_radius+1, x:x+2*patch_radius+1]
                G_patch = G_padded_image[y:y+2*patch_radius+1, x:x+2*patch_radius+1]
                B_patch = B_padded_image[y:y+2*patch_radius+1, x:x+2*patch_radius+1]                           
                intensity_value = image[y, x]
                mean = np.mean(patch)
                median = np.median(patch)
                modd = get_all_modes(patch.flatten())
                skew_val, kurt_val=ske_kur(patch)  # Set bias=False for sample (not population)
                # dc=calculate_degree_centrality(patch)
                #v_t = round(scaled_sigmoid(x, max_varigram, 0, L, k, x0), 2)
                fl_pix_val = fl_math(intensity_value, mean, median, modd, sv1,sv2)
                #new_pix_val = fl_math(intensity_value,variogram,std_deviation, mean, median, modd, skew_val, kurt_val, adj_shift, dc) 
                if(fl_pix_val<a):
                    new_img[y,x] = 0 
                elif (fl_pix_val >=a and fl_pix_val <=140):                   
                    adj_shift_by_std=round(adjacency_shift(patch, 0)/calculate_standard_deviation(patch),2)
                    vario_std = round(calculate_variogram(patch)/calculate_standard_deviation(patch),2)
                    #print("Normalized Variogram:**********************************", vario_std)
                    if vario_std <sv4:
                        #return 0  # Homogeneous → background
                        moran_i= calculate_morans_i(patch)
                        if moran_i<sv5:
                            new_img[y,x]=0
                        else:
                            R_dnsi_by_std=round(adjacency_shift(R_patch, 0)/calculate_standard_deviation(R_patch),2)
                            G_dnsi_by_std=round(adjacency_shift(G_patch, 0)/calculate_standard_deviation(G_patch),2)
                            B_dnsi_by_std=round(adjacency_shift(B_patch, 0)/calculate_standard_deviation(B_patch),2)
                            new_img[y,x] = vote(intensity_value,fl_pix_val,vario_std,R_dnsi_by_std,G_dnsi_by_std,B_dnsi_by_std,img_colr_R[y,x],img_colr_G[y,x],img_colr_B[y,x])
                            ccc+=1
                    else:
                        R_dnsi_by_std=round(adjacency_shift(R_patch, 0)/calculate_standard_deviation(R_patch),2)
                        G_dnsi_by_std=round(adjacency_shift(G_patch, 0)/calculate_standard_deviation(G_patch),2)
                        B_dnsi_by_std=round(adjacency_shift(B_patch, 0)/calculate_standard_deviation(B_patch),2)
                        new_img[y,x] = vote(intensity_value,fl_pix_val,vario_std,R_dnsi_by_std,G_dnsi_by_std,B_dnsi_by_std,img_colr_R[y,x],img_colr_G[y,x],img_colr_B[y,x])
                        ccc+=1
                else:
                    new_img[y,x] = 255 # np.random.randint(100, 211) #if new_pix_val >= 127 else 0
            elif new_img[y,x]<=a:
                new_img[y,x]=0 
            else:
                new_img[y,x]=0    
    print(ccc)
    #Eor_dia(new_img,output_file)
    
    cv2.imwrite(output_file,new_img)
    display_matplt(image, new_img)




        



















def main():
    # Check if the correct number of arguments is provided
    excel_file="tool.xlsx"
    image_path="C:/Users/Jit/AppData/Local/Programs/Python/Python312/z_My_Work/CV/All_IMGs/3411.jpg"
    #image_path="/home/surajit/CV/Unsupervised_Learning_NEW/341.jpg"
    color_spec='Perceived Grayscale Analysis'
    neighborhood_size=7
    global sv1,sv2,sv3,sv4,sv5
    
  
    if len(sys.argv) != 10:
        print("Usage: python imaproc.py <excel_file>", sys.argv[1])
        sys.exit(1)
    excel_file = sys.argv[1]
    image_path = sys.argv[2]
   
    color_spec=sys.argv[3]
    neighborhood_size=int(sys.argv[4])
    
    sv1=float(sys.argv[5]) 
    sv2=float(sys.argv[6])
    sv3=float(sys.argv[7])
    sv4=float(sys.argv[8])
    sv5=float(sys.argv[9])
    #flg_hough=float(sys.argv[10])

    #print("sv3:",sv3,"sv4:",sv4,"sv5:",sv5)
  
    
    
    
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))

    # Create output file name
    output_file = f"0_{file_name}_{neighborhood_size}{file_extension}"
    

    

         
    
    image_original = cv2.imread(image_path)
    if image_original is None:
        print("Error: Image could not be loaded.")
    else:
        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)

        height,width,_=image_original.shape
         
        R = image_original[:,:,0].astype(float)
        G = image_original[:,:,1].astype(float)
        B = image_original[:,:,2].astype(float)
        image = np.zeros((height, width, 1), dtype=np.uint8)  
        #image = np.zeros((height, width), dtype=np.uint8)    
         
        if color_spec=="HIP Perceived Grayscale Analysis":
             image = np.round(0.299 * R + 0.587 * G + 0.114 * B)
             #image = image.astype(np.float32)
        else:
             image=np.round(R/3+G/3+B/3)  



        # Print the size of the input image (height, width, and channels)
        height, width = image.shape
        #print(f"********Input image size: Height = {height}, Width = {width}")

  
        
        #print(f"Excel file: {excel_file}")
        
        
        
        tet=read_sample(excel_file, image, neighborhood_size)
        

        #new_img = np.zeros_like(trans_img, dtype=np.uint8)
        #new_img = np.zeros((height, width, 1), dtype=np.uint8)

        # Extract spatial features, classify regions, and save data to Excel
        
       

        extract_and_classify_image_gray(image_original,image, output_file, neighborhood_size,sv1, sv2)
        #process_image_with_progress(image, output_file, neighborhood_size)


    #input("Press Enter to exit...")
    
    

if __name__ == "__main__":
    main()
    #cProfile.run('main()') #main()





