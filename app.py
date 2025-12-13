import streamlit as st
import torch
import torch.nn as nn
import torchio as tio
import os
import tempfile
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeuroClarity AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. DEFINE THE MODEL ARCHITECTURE ---
# (Must match the training code exactly)
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Visual Branch (3D CNN)
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Flatten()
        )
        self.cnn_fc = nn.Linear(16384, 64)
        
        # Clinical Branch (MLP)
        self.mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64+16, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )
    
    def forward(self, img, clinical):
        x1 = self.cnn_fc(self.cnn(img))
        x2 = self.mlp(clinical)
        return self.fusion(torch.cat((x1, x2), dim=1))

# --- 2. LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model():
    device = torch.device("cpu") # GitHub/Streamlit Cloud usually uses CPU
    model = FusionModel()
    
    # Check if model file exists
    model_path = "NeuroClarity_3D_v1.pth"
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file '{model_path}' not found! Please upload it to your GitHub repository.")
        return None
    
    try:
        # Load state dict
        # We use map_location='cpu' to ensure it works even if trained on GPU
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle DataParallel keys (remove 'module.' prefix if present)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# --- 3. PREDICTION FUNCTION ---
def predict(age, gender, mmse, mri_file):
    if model is None:
        return "Model Error", 0, 0

    # A. Process Clinical Data
    gender_val = 1.0 if gender == 'Female' else 0.0
    # Normalize: Age/100, Gender 0/1, MMSE/30
    clinical_tensor = torch.tensor([[age/100.0, gender_val, mmse/30.0]], dtype=torch.float32)

    # B. Process MRI
    # Save uploaded file temporarily because TorchIO needs a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(mri_file.getbuffer())
        tmp_path = tmp.name

    try:
        # Preprocessing transforms (Must match training!)
        transforms = tio.Compose([
            tio.RescaleIntensity((0, 1)),
            tio.Resize((64, 64, 32))
        ])
        
        subject = tio.Subject(mri=tio.ScalarImage(tmp_path))
        subject = transforms(subject)
        img_tensor = subject.mri.data.unsqueeze(0) # Add batch dimension [1, 1, 64, 64, 32]
        
        # Cleanup temp file
        os.unlink(tmp_path)

        # C. Inference
        with torch.no_grad():
            output = model(img_tensor, clinical_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            
            prob_healthy = probs[0][0].item() * 100
            prob_dementia = probs[0][1].item() * 100
            
            return ("Dementia" if prob_dementia > prob_healthy else "Healthy"), prob_healthy, prob_dementia

    except Exception as e:
        return f"Error: {e}", 0, 0

# --- 4. STREAMLIT UI LAYOUT ---
st.title("🧠 NeuroClarity: Alzheimer's Early Detection")
st.markdown("""
**Complex Engineering Problem (CEP) - AI Lab Project** *A Multimodal Fusion System combining 3D MRI Scans + Clinical Demographics.*
""")

# Create two columns for input
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("1. Patient Demographics")
    age = st.slider("Patient Age", 40, 100, 70)
    gender = st.radio("Gender", ["Male", "Female"])
    
    st.info("💡 **MMSE Score:** Mini-Mental State Exam (0-30). Lower scores indicate cognitive impairment.")
    mmse = st.slider("MMSE Score", 0, 30, 24)

with col2:
    st.header("2. Neuroimaging Data")
    uploaded_file = st.file_uploader("Upload 3D MRI Scan (.nii.gz)", type=["nii", "gz"])
    
    if uploaded_file:
        st.success(f"File Uploaded: {uploaded_file.name}")
        st.markdown("✅ *3D Volume Ready for Processing*")
    else:
        st.warning("Please upload a .nii.gz file to proceed.")

# --- 5. RUN DIAGNOSIS ---
st.markdown("---")
if st.button("🚀 Run NeuroClarity Diagnosis", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.error("⚠️ Please upload an MRI scan first.")
    else:
        with st.spinner("Processing 3D Brain Volume & fusing with Clinical Data..."):
            result, p_healthy, p_dementia = predict(age, gender, mmse, uploaded_file)
        
        # Display Results
        st.markdown("### 📋 Diagnostic Report")
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Condition", result, delta_color="inverse")
        m2.metric("Dementia Probability", f"{p_dementia:.2f}%")
        m3.metric("Healthy Probability", f"{p_healthy:.2f}%")
        
        # Visual Bar
        st.progress(int(p_dementia))
        
        if result == "Dementia":
            st.error("🚨 **High Risk Detected:** The Fusion Model indicates patterns consistent with Alzheimer's Dementia.")
        elif str(result).startswith("Error"):
            st.error(result)
        else:
            st.success("✅ **Low Risk:** The model predicts the patient is Cognitively Normal.")

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("About NeuroClarity")
    st.markdown("""
    This system uses a **Fusion Architecture**:
    - **Input A:** 3D CNN (Brain Scan)
    - **Input B:** MLP (Demographics)
    
    **Accuracy:** 81.57% (Test Set)
    
    **Engineering Principles:**
    - EP1: Multimodal Deep Learning
    - EP2: Handling Class Imbalance
    - EP7: Sustainable AI (T4 Optimized)
    """)
    st.markdown("---")
    st.caption("Developed by Group XX for CSE-412.")
