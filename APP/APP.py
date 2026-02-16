import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
import datetime

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("wafer_model.keras")
print("Model Outputs:", model.outputs)
print("Number of outputs:", len(model.outputs))


class_names = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-Full", "Random", "Scratch"
]

history_log = []

# =========================
# GET INPUT SIZE SAFELY
# =========================
def get_model_input_size(model):
    shape = model.inputs[0].shape
    return int(shape[1]), int(shape[2])

# =========================
# FIND LAST CONV LAYER
# =========================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

LAST_CONV_LAYER = get_last_conv_layer(model)
print("Last Conv Layer:", LAST_CONV_LAYER)

# =========================
# GRADCAM
# =========================
def make_gradcam_heatmap(img_array):
    if LAST_CONV_LAYER is None:
        return np.zeros((512, 512))

    # If model has multiple outputs, take first output
    model_output = model.output
    if isinstance(model_output, list):
        model_output = model_output[0]

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model_output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])

        if isinstance(predictions, list):
            predictions = predictions[0]

        # Ensure predictions is (num_classes,)
        predictions = tf.squeeze(predictions)

        pred_index = tf.argmax(predictions)
        class_channel = predictions[pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    if tf.reduce_max(heatmap) == 0:
        return np.zeros((512, 512))

    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()


# =========================
# PDF GENERATION
# =========================
def generate_pdf():
    if len(history_log) == 0:
        return None

    file_path = "wafer_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    elements = []

    style = ParagraphStyle(
        name="Normal",
        fontSize=12,
        textColor=colors.black
    )

    elements.append(Paragraph("<b>Wafer AI Inspection Report</b>", style))
    elements.append(Spacer(1, 12))

    for entry in history_log:
        text = f"""
        Time: {entry['Time']} <br/>
        Defect: {entry['Defect']} <br/>
        Confidence: {entry['Confidence']:.2f}
        """
        elements.append(Paragraph(text, style))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return file_path

# =========================
# ANALYZE FUNCTION
# =========================
def analyze_single(image):
    try:
        if image is None:
            return "Upload Image", 0, "No Data", None, None, pd.DataFrame()

        original = image.copy()

        height, width = get_model_input_size(model)

        image = image.convert("RGB")
        image = image.resize((width, height))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict([img_array], verbose=0)
        prediction = prediction[0]

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        if confidence > 0.75:
            reliability = "🟢 High"
        elif confidence > 0.60:
            reliability = "🟡 Moderate"
        else:
            reliability = "🔴 Low.Check It Manually."

        heatmap = make_gradcam_heatmap(img_array)

        original_resized = np.array(original.resize((512, 512)))
        heatmap_resized = cv2.resize(heatmap, (512, 512))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        result_img = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)

        history_log.append({
            "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Defect": predicted_class,
            "Confidence": confidence
        })

        df_history = pd.DataFrame(history_log)

        return (
            predicted_class,
            confidence,
            reliability,
            result_img,
            None,
            df_history
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"Error: {str(e)}",
            0,
            "Failed",
            None,
            None,
            pd.DataFrame()
        )

# =========================
# GRADIO UI
# =========================
with gr.Blocks() as demo:

    gr.Markdown("""
    # 🚀 Smart Wafer AI Monitoring System  
    ### Industrial-Grade Defect Detection with Explainable AI
    """)

    with gr.Row():
        input_img = gr.Image(type="pil", label="Upload Wafer Image")
        analyze_btn = gr.Button("🔍 Analyze", variant="primary")

    with gr.Row():
        out_class = gr.Textbox(label="Predicted Defect")
        out_conf = gr.Number(label="Confidence Score")
        out_rel = gr.Textbox(label="Reliability Level")

    gradcam_img = gr.Image(label="High-Resolution Grad-CAM (512px)")
    history_table = gr.Dataframe(label="Inspection History")

    with gr.Row():
        pdf_btn = gr.Button("📄 Download Inspection Report")
        pdf_file = gr.File()

    analyze_btn.click(
        analyze_single,
        inputs=input_img,
        outputs=[
            out_class,
            out_conf,
            out_rel,
            gradcam_img,
            pdf_file,
            history_table
        ]
    )

    pdf_btn.click(generate_pdf, outputs=pdf_file)

demo.launch()
