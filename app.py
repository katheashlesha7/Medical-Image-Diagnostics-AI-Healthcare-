"""Optional local Streamlit interface for the trained educational model."""

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from cxr.model import build_model, make_transforms


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "models" / "best.pt"

st.set_page_config(page_title="Chest X-ray ML Demo", page_icon="🩻", layout="centered")
st.title("Chest X-ray classification demo")
st.warning(
    "Educational research demo only. It is not a medical device and must not be used "
    "to diagnose, rule out disease, or guide care. Consult a qualified clinician."
)
st.write(
    "Upload one chest X-ray to view the model's scores for the 14 NIH ChestX-ray14 labels. "
    "Labels may overlap. Scores are not calibrated clinical probabilities."
)


@st.cache_resource
def load_checkpoint_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    labels = checkpoint["class_labels"]
    model = build_model(len(labels), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, labels, device, checkpoint.get("threshold", 0.5)


if not CHECKPOINT_PATH.exists():
    st.info("No trained checkpoint found. Run the training command in the README first.")
else:
    uploaded = st.file_uploader("Choose a PNG or JPEG image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Show model scores", type="primary"):
            model, labels, device, threshold = load_checkpoint_model(str(CHECKPOINT_PATH))
            tensor = make_transforms()(image).unsqueeze(0).to(device)
            with torch.no_grad():
                scores = torch.sigmoid(model(tensor))[0].cpu().tolist()
            ordered = sorted(zip(labels, scores), key=lambda item: item[1], reverse=True)
            st.subheader("Model scores")
            for label, score in ordered:
                st.write(f"**{label.replace('_', ' ')}** — {score:.3f}")
            st.caption(
                f"The demo cutoff is {threshold:.2f}; it has not been clinically calibrated. "
                "A score above it is not a diagnosis."
            )
