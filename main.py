import cv2
import streamlit as st
import numpy as np
import pandas as pd
old_1 = 1
old_2 = 1
old_3 = 1
old_1a = 1
old_2a = 1
old_3a = 1
left_lane_count = 0
middle_lane_count = 0
right_lane_count = 0
left_lane_counta = 0
middle_lane_counta = 0
right_lane_counta = 0

def map_plot():
    df = pd.DataFrame(np.random.randn(3, 2) / [850, 850] + [12.823507946143955, 80.04369223216773],columns=['lat', 'lon'])
    st.map(df, zoom=15)

def line_plot():
    chart_data = pd.DataFrame({
        'Incomming Traffic': [10,20,30,40,50,23,54,14,16,32],
        'Outgoing Traffic': [30,10,50,90,70,65,34,76,12,11]
    },
    columns=['Incomming Traffic', 'Outgoing Traffic'])
    st.line_chart(chart_data,width=100, height=250)




def main(t, threshold_valA, threshold_valB):


    FRAME_WINDOW = st.image([])

    from cvzone import stackImages
    cap = cv2.VideoCapture(r'pexels-kelly-lacy-5473765.mp4')
    capa = cv2.VideoCapture(r'tt.mp4')
    if (cap.isOpened() == False):
        print("Error opening video stream or file")   





    def coutnCarLeft(pixalVal):
        global old_1
        global left_lane_count
        new = pixalVal
        if new == 200 and old_1 == 0:
            left_lane_count += 1
        old_1 = new
    def coutnCarMiddle(pixalVal):
        global old_2
        global middle_lane_count
        new = pixalVal
        if new == 200 and old_2 == 0:
            middle_lane_count += 1
        old_2 = new
    def coutnCarRight(pixalVal):
        global old_3
        global right_lane_count
        new = pixalVal
        if new == 200 and old_3 == 0:
            right_lane_count += 1
        old_3 = new
    def coutnCarLefta(pixalVal):
        global old_1a
        global left_lane_counta
        newa = pixalVal
        if newa == 200 and old_1a == 0:
            left_lane_counta += 1
        old_1a = newa
    def coutnCarMiddlea(pixalVal):
        global old_2a
        global middle_lane_counta
        newa = pixalVal
        if newa == 200 and old_2a == 0:
            middle_lane_counta += 1
        old_2a = newa
    def coutnCarRighta(pixalVal):
        global old_3a
        global right_lane_counta
        newa = pixalVal
        if newa == 200 and old_3a == 0:
            right_lane_counta += 1
        old_3a = newa

    if b.button("Start"):
        b.button("Stop")
        while (cap.isOpened()):
            ret, frame = cap.read()
            __, framea = capa.read()
            if ret == True:
                frame = cv2.resize(frame, (960, 540))
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_canny = cv2.Canny(img_gray, 100, 100)
                _, threshImg = cv2.threshold(img_gray, 50, 200, cv2.THRESH_BINARY)

                framea = cv2.resize(framea, (960, 540))
                img_graya = cv2.cvtColor(framea, cv2.COLOR_BGR2GRAY)
                img_cannya = cv2.Canny(img_graya, 100, 100)
                _a, threshImga = cv2.threshold(img_graya, 50, 200, cv2.THRESH_BINARY)

                # cv2.circle(frame, (267, 496), 5, (255, 0, 0), 5)
                # cv2.circle(frame, (475, 501), 5, (255, 0, 0), 5)
                # cv2.circle(frame, (684, 498), 5, (255, 0, 0), 5)
                coutnCarLeft(threshImg[496][267])
                coutnCarMiddle(threshImg[501][475])
                coutnCarRight(threshImg[498][684])


                # cv2.circle(framea, (370, 480), 2, (0, 0, 0), 10)
                # cv2.circle(framea, (590, 480), 2, (0, 0, 0), 10)
                # cv2.circle(framea, (800, 480), 2, (0, 0, 0), 10)
                coutnCarLefta(threshImga[480][370])
                coutnCarMiddlea(threshImga[480][590])
                coutnCarRighta(threshImga[480][800])

                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(frame, f"Total Vehicle :  {left_lane_count+middle_lane_count+right_lane_count}",
                #            (35, 35), font, .6, (0, 255, 0), 2)
                # cv2.putText(framea, f"Total Vehicle :  {left_lane_counta+middle_lane_counta+right_lane_counta}",
                #            (35, 35), font, .6, (0, 255, 0), 2)

        
                total_a = left_lane_count+middle_lane_count+right_lane_count
                total_b = left_lane_counta+middle_lane_counta+right_lane_counta
        
                frame = cv2.circle(frame, (25,25), 10, (0,255,0),15)

                framea = cv2.circle(framea, (25,25), 10, (0,255,0),15)

                if threshold_valA <= total_a:
                    frame = cv2.circle(frame, (25,25), 10, (255,0,0),15)
                    cv2.putText(frame, f"Vehicle exceded threshold value ie : 11",(25, 85), font, 1.2, (255,255, 0), 2)

                if threshold_valB <= total_b:
                    framea = cv2.circle(framea, (25,25), 10, (255,0,0),15)
                if threshold_valA > total_a:
                    frame = cv2.circle(frame, (25,25), 10, (0,255,0),15)
                if threshold_valB > total_b:
                    framea = cv2.circle(framea, (25,25), 10, (0,255,0),15)
                
                if total_a == 13:
                    cv2.waitKey(-1)
            


                frameb = stackImages([frame, framea], 2, 0.8)
                t.subheader(f"Vehicle: Entry_A = {total_a} & Entry_B = {total_b}")
                FRAME_WINDOW.image(frameb)

        cap.release()
            
        cv2.destroyAllWindows()
    else:
        st.caption("Press the button to start streaming")

if __name__ == '__main__':

    st.title("Traffic Management")

    

    t = st.empty()
    b = st.empty()
    threshold_valA = 11
    threshold_valB = 21
    main(t, threshold_valA, threshold_valB)

    st.header("Density of vehicle")
    map_plot()
    st.header("Entry and exits of vehicles")
    st.caption("It'll be further used to train the model and find a better solution")
    line_plot()

    st.subheader("Roadmap of SRM KTR")
    st.image(r'pic.png')
    
    # main(t, threshold_valA, threshold_valB)