import streamlit as st
import pandas as pd
import cv2
from streamlit_drawable_canvas import st_canvas
from src import *

@st.cache_resource
def get_model():
    return load_model()

parameters = get_model()

st.title("Digit Recognizer")
st.write("Draw a digit between 0 and 9")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    processed, preview = prepare_canvas_image(canvas_result.image_data)
    if processed is None:
        st.info("Please draw a digit on the canvas.")
        st.stop()
    prediction, confidence = predict_one(processed, parameters)
    big_preview = cv2.resize(preview, (280, 280), interpolation=cv2.INTER_NEAREST)
    st.subheader("What the Model Sees")
    st.image(big_preview)

    if confidence[prediction] < 0.60:
        st.warning("Low confidence prediction. Try drawing more clearly.")

    st.header(f"Prediction: {prediction}")
    st.write(f"Confidence: {confidence[prediction] * 100:.2f}%")
    st.write(f"Probability Sum: {confidence.sum():.4f}")

    chart_data = pd.DataFrame(
        {"Probability": confidence},
        index=[str(i) for i in range(10)]
    )

    st.subheader("Probabilities")
    st.bar_chart(chart_data)

if st.button("Clear Canvas"):
    st.rerun()