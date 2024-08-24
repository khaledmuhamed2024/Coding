import os
import scipy.io
import numpy as np
import pywt
from scipy.signal import welch

def load_mat_file(file_path):
    try:
        mat_data = scipy.io.loadmat(file_path)
        
        # Check if 'o' key exists in the loaded data
        if 'o' not in mat_data:
            raise ValueError(f"No 'o' key found in {file_path}")
        
        o_data = mat_data['o']
        
        # Ensure o_data is a numpy ndarray and has expected structure
        if not isinstance(o_data, np.ndarray) or o_data.shape[0] == 0:
            raise ValueError(f"Unexpected structure in 'o' data of {file_path}")
        
        # Extract data from the first element of the o_data array
        o_data_first_elem = o_data[0, 0]
        
        # Extract the specific fields from the structured array
        signal_id = o_data_first_elem[0][0]
        signal_data = o_data_first_elem[4]
        
        return signal_data
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_signal_data(signal_data):
    # Example preprocessing: normalize the signal data
    mean_val = np.mean(signal_data)
    std_val = np.std(signal_data)
    
    normalized_data = (signal_data - mean_val) / std_val
    
    return normalized_data

def extract_features_psd(signal_data, fs=256):
    # Compute the Power Spectral Density (PSD) using Welch's method
    freqs, psd = welch(signal_data, fs=fs, nperseg=1024)
    return psd

def extract_features_wavelet(signal_data, wavelet='db4', level=4):
    # Extract Discrete Wavelet Transform (DWT) features
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    features = np.concatenate([np.ravel(c) for c in coeffs])
    return features

def save_extracted_features(file_path, psd_features, wavelet_features, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Construct the output file path
    base_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, base_name)
    
    # Save the extracted features to the output file
    scipy.io.savemat(output_file_path, {
        'psd_features': psd_features,
        'wavelet_features': wavelet_features
    })

def process_all_files(input_dir, output_dir):
    # Iterate through all .mat files in the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mat'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                
                # Load the .mat file
                signal_data = load_mat_file(file_path)
                
                if signal_data is not None:
                    # Preprocess the signal data
                    preprocessed_data = preprocess_signal_data(signal_data)
                    
                    # Extract features using PSD and Wavelet Transform
                    psd_features = extract_features_psd(preprocessed_data)
                    wavelet_features = extract_features_wavelet(preprocessed_data)
                    
                    # Save the extracted features to the output directory
                    save_extracted_features(file_path, psd_features, wavelet_features, output_dir)
                    print(f"Saved extracted features for {file} to {output_dir}")
                else:
                    print(f"Failed to process {file_path}")

# Input directory containing the .mat files
input_dir = r'D:\org\BrainStorming\projects\EEG data for Mental Attention\archive\EEG Data\EEG Data'

# Output directory to save the extracted features
output_dir = r'D:\org\BrainStorming\projects\EEG data for Mental Attention\archive\EEG Data\processed'

# Process all files in the input directory and save the extracted features to the output directory
process_all_files(input_dir, output_dir)
