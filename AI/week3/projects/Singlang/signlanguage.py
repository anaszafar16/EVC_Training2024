import cv2
import numpy as np
from ultralytics import YOLO
import base64
import streamlit as st


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def main():
    names_count = {"yes": 0, "hello": 0, "thanks": 0}
    yolo = YOLO("model\\bestV3.pt")
    st.set_page_config(layout="wide", page_title="Sign",
                       page_icon="images\\sllogof.png")

    # st.sidebar.title("Sign Connect")
    logo = "images\\sllogof.png"
    st.sidebar.image(logo)
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #e4d4c8 ;
        }
    </style>
    """, unsafe_allow_html=True)


    st.caption("Powered by OpenCV, Streamlit")
    cap = cv2.VideoCapture(0)
    c1, c2,  = st.columns([5, 1])
    with c1:
        st.markdown("---")
        frame_placeholder = st.empty()
        st.markdown("---")
        stop_button_pressed = st.button("Stop")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")

    with c2:
        st.markdown("---")
        yes = st.empty()
        st.markdown("---")
        hello = st.empty()
        st.markdown("---")
        thanks = st.empty()
        st.markdown("---")


    def post_process(detections):
        for detection in detections:
            if len(detection.boxes.xyxy) == 0:
                return None, None
            else:
                bbox = detection.boxes.xyxy[0].cpu() if detection.boxes.xyxy[0] is not None else np.empty((0, 4))
                bbox = bbox.numpy().astype(int)

                class_index = int(detection.boxes[0].cls)
                class_name = yolo.names[class_index]

                return bbox , class_name

    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()

        det = yolo.predict(frame, imgsz=448)
        bbox , name = post_process(det)
        if name is not None:
            names_count[name] += 1
        color = (0, 255, 0)
        if bbox is not None:
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, name, (bbox[0], bbox[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if not ret:
            st.write("Video Capture Ended")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", width=1200)
        yes.markdown(f'yes : {names_count["yes"]}')
        hello.markdown(f'hello : {names_count["hello"]}')
        thanks.markdown(f' thanks : {names_count["thanks"]}')
        if names_count["yes"] == 5:

            autoplay_audio("voices/yes 1.wav")
            names_count = {"yes": 0, "hello": 0, "thanks": 0}
        if names_count["hello"] == 5:

            autoplay_audio("voices/hello 1.wav")
            names_count = {"yes": 0, "hello": 0, "thanks": 0}
        if names_count["thanks"] == 5:

            autoplay_audio("voices/thanks 1.wav")
            names_count = {"yes": 0, "hello": 0, "thanks": 0}

        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    cap.release()
    cv2.destroyAllWindows()


main()
