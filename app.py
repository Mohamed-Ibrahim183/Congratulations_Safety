import streamlit as st
import os
from ultralytics import YOLO
import tempfile
import numpy as np
from PIL import Image

# Set page config for modern look
st.set_page_config(
    page_title="YOLO Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stat-box {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem;
        color: #2c3e50;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Main header
st.markdown(
    '<div class="main-header">🔍 YOLO Object Detection Dashboard</div>',
    unsafe_allow_html=True,
)

# Sidebar for inputs
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)

    # List models in the models folder
    models_dir = "models/"
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    else:
        models = []

    if not models:
        st.error("No models found in the models folder.")
        st.stop()

    model_choice = st.selectbox(
        "🎯 Select Model", models, help="Choose a YOLO model from the models folder"
    )

    # Load model to get info
    model_path = os.path.join(models_dir, model_choice)
    model = YOLO(model_path)

    # Display model info
    st.subheader("📋 Model Information")
    st.write(f"**Classes:** {len(model.names)}")
    with st.expander("View Class Names"):
        st.write(list(model.names.values()))

    # Confidence threshold
    conf_threshold = st.slider(
        "🎚️ Confidence Threshold",
        0.1,
        1.0,
        0.5,
        0.05,
        help="Minimum confidence for detections",
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Image or Video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        help="Supported formats: JPG, PNG for images; MP4, AVI, MOV for videos",
    )

    # Run button
    run_button = st.button("🚀 Run Detection", type="primary", use_container_width=True)

# Main content area
if run_button:
    if uploaded_file is not None:
        with st.spinner("🔄 Processing... Please wait."):
            # Determine if it's an image or video
            if uploaded_file.type.startswith("image"):
                # Process image
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".jpg"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name

                results = model(tmp_path, conf=conf_threshold)
                annotated_img = results[0].plot()

                # Display results
                st.subheader("🖼️ Detection Results")

                # Create columns for original and annotated
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.image(
                        Image.open(tmp_path),
                        caption="Original Image",
                        width="content",
                        # use_column_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.image(annotated_img, caption="Annotated Image", width="content")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Statistics
                detections = len(results[0].boxes)
                st.subheader("📊 Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f'<div class="stat-box"><strong>Objects Detected:</strong><br>{detections}</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    classes_detected = (
                        set(results[0].boxes.cls.int().tolist())
                        if detections > 0
                        else set()
                    )
                    st.markdown(
                        f'<div class="stat-box"><strong>Classes Detected:</strong><br>{len(classes_detected)}</div>',
                        unsafe_allow_html=True,
                    )
                with col3:
                    avg_conf = (
                        results[0].boxes.conf.mean().item() if detections > 0 else 0
                    )
                    st.markdown(
                        f'<div class="stat-box"><strong>Avg Confidence:</strong><br>{avg_conf:.2f}</div>',
                        unsafe_allow_html=True,
                    )

                # Download button for annotated image
                annotated_pil = Image.fromarray(annotated_img)
                buf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                annotated_pil.save(buf.name)
                buf.close()
                with open(buf.name, "rb") as file:
                    data = file.read()
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=data,
                    file_name="annotated_image.jpg",
                    mime="image/jpeg",
                )
                os.unlink(buf.name)

                # Clean up temp file
                os.unlink(tmp_path)

            elif uploaded_file.type.startswith("video"):
                # Process video
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name

                # Run detection on video
                results = model(tmp_path, save=True, conf=conf_threshold)

                # Find the output video path
                output_dir = "runs/detect/predict/"
                if os.path.exists(output_dir):
                    basename = os.path.basename(tmp_path).split(".")[0]
                    output_files = [
                        f
                        for f in os.listdir(output_dir)
                        if f.startswith(basename)
                        and f.endswith((".mp4", ".avi", ".mov"))
                    ]
                    st.write(f"Output files found: {output_files}")  # Debug
                    if output_files:
                        output_video = os.path.join(output_dir, output_files[0])
                        st.write(f"Selected video: {output_video}")  # Debug
                        st.write(
                            f"File exists: {os.path.exists(output_video)}"
                        )  # Debug
                        # Use relative path for st.video
                        relative_path = os.path.relpath(output_video, os.getcwd())
                        st.write(f"Relative path: {relative_path}")  # Debug
                        st.subheader("🎥 Detection Results")
                        st.video(relative_path)
                        st.success(
                            "✅ Detection completed. Annotated video displayed above."
                        )

                        # Statistics (approximate from results)
                        total_detections = (
                            sum(len(r.boxes) for r in results) if results else 0
                        )
                        st.subheader("📊 Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(
                                f'<div class="stat-box"><strong>Total Objects Detected:</strong><br>{total_detections}</div>',
                                unsafe_allow_html=True,
                            )
                        with col2:
                            avg_conf = (
                                np.mean(
                                    [
                                        r.boxes.conf.mean().item()
                                        for r in results
                                        if len(r.boxes) > 0
                                    ]
                                )
                                if results and any(len(r.boxes) > 0 for r in results)
                                else 0
                            )
                            st.markdown(
                                f'<div class="stat-box"><strong>Avg Confidence:</strong><br>{avg_conf:.2f}</div>',
                                unsafe_allow_html=True,
                            )

                        # Download button for annotated video
                        with open(output_video, "rb") as file:
                            data = file.read()
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=data,
                            file_name="annotated_video.mp4",
                            mime="video/mp4",
                        )
                    else:
                        st.error("Annotated video not found.")
                else:
                    st.error("Output directory not found.")

                # Clean up temp file
                os.unlink(tmp_path)
            else:
                st.error("❌ Unsupported file type.")
    else:
        st.error("❌ Please upload a file.")
else:
    # Welcome message
    st.info(
        "👋 Welcome! Select a model, upload a file, and click 'Run Detection' to get started."
    )

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and YOLO")
