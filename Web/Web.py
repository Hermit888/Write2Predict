import streamlit as st
from streamlit_drawable_canvas import st_canvas

from process_image import process_image
from predict_image import predict

st.title("Write2Predict")
st.caption("Select a digit or a letter below, then draw it on the canvas")
st.caption("['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']")

canvas_result = st_canvas(
    stroke_width = 30,
    stroke_color='#FFFFFF',
    background_color= 'rgb(0, 0, 0)',
    update_streamlit=True,
    height=420, # 28 * 15
    width=420
)

if st.button('Predict'):
    if canvas_result.image_data is not None:
        input = process_image(canvas_result)

        # pop up error messgae if there is no content
        if input is None:
            st.write("Please draw a digit or a letter on canvas")
        else:
            pred = predict(input)

            st.write("Predicted character: ", pred)