import streamlit as st
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="AuraRad Portfolio", layout="wide")

st.title("3D Brain Tumor Radiomics & Volumetric Analytics Terminal")
st.write("Demonstrating end-to-end multi-modal semantic segmentation with a GPU-optimized 3D U-Net pattern.")

@st.cache_resource
def load_production_pipeline():
    network = torch.jit.load("brain_tumor_unet3d.pt")
    network.eval()
    meta = joblib.load("unet3d_pipeline_metadata.pkl")
    return network, meta

try:
    model, metadata = load_production_pipeline()
    voxel_scale = metadata["spatial_resolution_mm3"]
    spatial_shape = metadata["tensor_input_shape"]
except FileNotFoundError:
    st.error("Deployment Error: Make sure 'brain_tumor_unet3d.pt' and 'unet3d_pipeline_metadata.pkl' are in your folder.")
    st.stop()

if "active_scan" not in st.session_state:
    np.random.seed(2026)
    raw_telemetry = np.random.normal(loc=105.0, scale=12.0, size=(1, *spatial_shape)).astype(np.float32)
    
    z_c, y_c, x_c = 16, 16, 16
    radius = 5
    zz, yy, xx = np.ogrid[:32, :32, :32]
    target_zone = ((zz - z_c)**2 + (yy - y_c)**2 + (xx - x_c)**2) <= radius**2
    raw_telemetry[0][target_zone] += 50.0
    
    st.session_state.active_scan = raw_telemetry

col_inputs, col_outputs = st.columns([1, 1.4])

with col_inputs:
    st.subheader("Pipeline Controls")
    slice_depth = st.slider("Axial Matrix Viewing Plane (Z-Axis)", 0, spatial_shape[0] - 1, 16)
    decision_threshold = st.slider("Probability Classification Threshold", 0.1, 0.9, 0.5, step=0.05)
    
    trigger_inference = st.button("RUN PIPELINE INFERENCE")
    
    if trigger_inference:
        raw_input_array = np.copy(st.session_state.active_scan)
        min_intensity = raw_input_array.min()
        max_intensity = raw_input_array.max()
        preprocessed_scan = (raw_input_array - min_intensity) / (max_intensity - min_intensity + 1e-8)
        
        input_tensor = torch.tensor(preprocessed_scan).unsqueeze(0)
        
        with torch.no_grad():
            output_logits = model(input_tensor)
            probability_map = torch.sigmoid(output_logits).squeeze(0).squeeze(0).numpy()
            
        binary_segmentation = (probability_map >= decision_threshold).astype(np.uint8)
        
        absolute_voxels = int(np.sum(binary_segmentation))
        calculated_volume_mm3 = absolute_voxels * voxel_scale
        
        st.markdown("---")
        st.markdown("### Operational Pipeline Analytics")
        st.metric(label="Calculated Lesion Volume", value=f"{calculated_volume_mm3:.2f} mm³")
        st.metric(label="Absolute Positive Voxel Count", value=f"{absolute_voxels} Pixels")
        
        st.session_state.computed_mask = binary_segmentation

with col_outputs:
    st.subheader("Dynamic Volumetric Visualization")
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor('none')
    
    axes[0].imshow(st.session_state.active_scan[0, slice_depth, :, :], cmap='bone')
    axes[0].set_title(f"Preprocessed Telemetry Slice #{slice_depth}", color='white', fontsize=10)
    axes[0].axis('off')
    
    if "computed_mask" in st.session_state:
        axes[1].imshow(st.session_state.active_scan[0, slice_depth, :, :], cmap='bone')
        mask_layer = st.session_state.computed_mask[slice_depth, :, :]
        
        rgba_overlay = np.zeros((*mask_layer.shape, 4))
        rgba_overlay[mask_layer == 1] = [0.89, 0.24, 0.24, 0.5]
        
        axes[1].imshow(rgba_overlay)
        axes[1].set_title("Segmented Structural Tissue Mask", color='#f87171', fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "Execute pipeline inference engine\nto render spatial label maps.", 
                     color='#9ca3af', ha='center', va='center', fontsize=10)
        axes[1].set_title("Target Segmentation Mask Output", color='white', fontsize=10)
    
    axes[1].axis('off')
    st.pyplot(fig, bbox_inches='tight')