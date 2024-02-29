import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
import cv2
import numpy as np
import pandas as pd
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image
import sqlite3
import torch
import PIL.Image
import pkg_resources as pkg
from streamlit_option_menu import option_menu



from ultralytics import YOLO
import queue
import sqlite3

import PIL.Image
import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



conn = sqlite3.connect('inventory_database.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS drinks (time TEXT, name TEXT, countt NUMERIC)')

c.execute('CREATE TABLE IF NOT EXISTS biscuits (time TEXT, name TEXT, countt NUMERIC)')

c.execute('CREATE TABLE IF NOT EXISTS shampoos (time TEXT, name TEXT, countt NUMERIC)')

c.execute('CREATE TABLE IF NOT EXISTS lays (time TEXT, name TEXT, countt NUMERIC)')

c.execute('CREATE TABLE IF NOT EXISTS Noodles (time TEXT, name TEXT, countt NUMERIC)')


def model():
    #return torch.hub.load('ultralytics/yolov5', 'custom', path="best1.pt",force_reload=True) 
    return torch.hub.load('WongKinYiu/yolov7', 'custom', 'best7.pt',force_reload=True, trust_repo=True)
    
    
    #return YOLO('best8.pt')

model=model()

class VideoProcessor:
    def __init__(self):
        self.res=None
        self.confidence=0.5

    def getRes(self):
        #time.sleep(5)
        return self.res

    def recv(self, frame):
 
        model.conf=self.confidence
        img = frame.to_ndarray(format="bgr24")
         
        # vision processing
        flipped = img[:, ::-1, :]

        # model processing
        im_pil = Image.fromarray(flipped)
        results = model(im_pil, size=112)
        self.res=results
        bbox_img = np.array(results.render()[0])

        
        return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")




#--client.showErrorDetails=false

if 'login' not in st.session_state:
   st.session_state.login = False



__login__obj = __login__(auth_token = "courier_auth_token",
                    company_name = "Shims",
                    width = 200, height = 250,
                    logout_button_name = 'Logout', hide_menu_bool = False,
                    hide_footer_bool = False,
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN= __login__obj.build_login_ui()


if 'LOGGED_IN' not in st.session_state:
    st.session_state.LOGGED_IN = False

def main():
        if LOGGED_IN == True:
            #st.markdown("Welcome to Main page!")

            st.title("Inventory Management System")
            #choice=st.selectbox("Mode",["None","Staff","Admin"])

            def get_opened_image(image):
                return Image.open(image)
            
            RTC_CONFIGURATION = RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            

                
            with st.sidebar:
                selected = option_menu(
                    menu_title=None,
                    options=["Image", "Data","Video"],
                    icons = ["image","bar-chart","camera-video"],
                    menu_icon = "list")
            #mode=st.sidebar.radio("View mode",['image','data','video'])

            if selected == "Image":
                st.subheader(f"{selected}")
                date1=st.date_input("Date")
                
                image_choice=st.selectbox("Mode",["Noodles","Shampoo","Lays","Cooldrinks","Biscuits"])
                confidence_threshold=st.slider("Confidence threshold",0.0,1.0)
                img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
                detect=st.checkbox("Show the detected labels")
                if detect:
                    if img_file_buffer is not None: 
                        image = np.array(PIL.Image.open(img_file_buffer)) 
                        image = cv2.resize(image, (640, 640))
                        #model=torch.hub.load('ultralytics/yolov5', 'custom', path='best1.pt')

                        
                        if image_choice=="Noodles":
                            model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'noodles.pt')
                        if image_choice=="Shampoo":
                            model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'best7.pt')
                        if image_choice=="Lays":
                            model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'lays.pt')
                        if image_choice=="Cooldrinks":
                            model=torch.hub.load('ultralytics/yolov5', 'custom', path='best1.pt')
                        if image_choice=="Biscuits":
                            model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'biscuits.pt')


                        #model = torch.hub.load('ultralytics/yolov5', 'custom', path="best1.pt") 
                        
                        #model = YOLO('best8.pt')

                        model.conf = confidence_threshold
                    
                        results=model(image)
                        
                        st.image(results.render()[0])
                        
                        
                        df=results.pandas().xyxy[0]
                        list_of_names=[]
                        for i in range(len(df)):
                                list_of_names.append(df.loc[i, "name"])
                        new_df=pd.DataFrame(list_of_names,columns=["count"])
                        st.sidebar.table(new_df["count"].value_counts())

                        if(st.button("STORE")):
                            if image_choice=="Cooldrinks":
                                for i in range(len(df)):
                                        c.execute('INSERT INTO drinks (time,name,countt) VALUES (?, ?, ?)',( date1, (df.loc[i, "name"]) , 1))
                                        conn.commit()
                                st.write("UPDATED DATABASE:")
                                c.execute('SELECT name,sum(countt) FROM drinks GROUP BY name ')
                                read_list3=c.fetchall()
                                d = pd.DataFrame(read_list3,columns=["name","count"])
                                st.table(d)
                            if image_choice=="Shampoo":
                                for i in range(len(df)):
                                        c.execute('INSERT INTO shampoos (time,name,countt) VALUES (?, ?, ?)',( date1, (df.loc[i, "name"]) , 1))
                                        conn.commit()
                                st.write("UPDATED DATABASE:")
                                c.execute('SELECT name,sum(countt) FROM shampoos GROUP BY name ')
                                read_list3=c.fetchall()
                                d = pd.DataFrame(read_list3,columns=["name","count"])
                                st.table(d)
                            if image_choice=="Lays":
                                for i in range(len(df)):
                                        c.execute('INSERT INTO lays (time,name,countt) VALUES (?, ?, ?)',( date1, (df.loc[i, "name"]) , 1))
                                        conn.commit()
                                st.write("UPDATED DATABASE:")
                                c.execute('SELECT name,sum(countt) FROM lays GROUP BY name ')
                                read_list3=c.fetchall()
                                d = pd.DataFrame(read_list3,columns=["name","count"])
                                st.table(d)
                            
                            if image_choice=="Biscuits":
                                for i in range(len(df)):
                                        c.execute('INSERT INTO biscuits (time,name,countt) VALUES (?, ?, ?)',( date1, (df.loc[i, "name"]) , 1))
                                        conn.commit()
                                st.write("UPDATED DATABASE:")
                                c.execute('SELECT name,sum(countt) FROM biscuits GROUP BY name ')
                                read_list3=c.fetchall()
                                d = pd.DataFrame(read_list3,columns=["name","count"])
                                st.table(d)

                            if image_choice=="Noodles":
                                for i in range(len(df)):
                                        c.execute('INSERT INTO noodles (time,name,countt) VALUES (?, ?, ?)',( date1, (df.loc[i, "name"]) , 1))
                                        conn.commit()
                                st.write("UPDATED DATABASE:")
                                c.execute('SELECT name,sum(countt) FROM noodles GROUP BY name ')
                                read_list3=c.fetchall()
                                d = pd.DataFrame(read_list3,columns=["name","count"])
                                st.table(d)
                else:
                    if img_file_buffer is not None:
                        image=get_opened_image(img_file_buffer)
                        with st.expander("Selected Image",expanded=True):
                            st.image(img_file_buffer,use_column_width=True)
            if selected=="Data":
                st.subheader("DATA")

                st.subheader("Categories")
                choice=st.selectbox("Mode",["All","Drinks","Lays","Shampoos","Biscuits","Noodles"])


                if choice=='All':

                    st.write("CoolDrinks DataBase")
                    read_list2=[]    
                    c.execute('SELECT time,name,sum(countt) FROM drinks GROUP BY name ')
                    read_list2=c.fetchall()
                    df2 = pd.DataFrame(
                        read_list2,
                        columns=["time","name","count"]
                    )
                    st.table(df2)

                    st.write("Biscuits DataBase")
                    read_list2=[]    
                    c.execute('SELECT time,name,sum(countt) FROM biscuits GROUP BY name ')
                    read_list2=c.fetchall()
                    df2 = pd.DataFrame(
                        read_list2,
                        columns=["time","name","count"]
                    )
                    st.table(df2)

                    st.write("Lays DataBase")

                    read_list2=[]    
                    c.execute('SELECT time,name,sum(countt) FROM lays GROUP BY name ')
                    read_list2=c.fetchall()
                    df2 = pd.DataFrame(
                        read_list2,
                        columns=["time","name","count"]
                    )
                    st.table(df2)

                    st.write("Shampoos DataBase")

                    read_list2=[]    
                    c.execute('SELECT time,name,sum(countt) FROM shampoos GROUP BY name ')
                    read_list2=c.fetchall()
                    df2 = pd.DataFrame(
                        read_list2,
                        columns=["time","name","count"]
                    )
                    st.table(df2)

                    st.write("Noodles DataBase")

                    read_list2=[]    
                    c.execute('SELECT time,name,sum(countt) FROM noodles GROUP BY name ')
                    read_list2=c.fetchall()
                    df2 = pd.DataFrame(
                        read_list2,
                        columns=["time","name","count"]
                    )
                    st.table(df2)
                     

                     
                elif choice=='Drinks':

                    drinkname=st.text_input("Enter drink name")
                    drinkcount=st.text_input("Enter count")
                    date=st.date_input("Select date")
                    

                    if(st.button("SUBMIT")):
                            c.execute('INSERT INTO drinks (time,name,countt) VALUES (?, ?, ?)',( date, drinkname , drinkcount))
                            conn.commit()
                    
                    read_list1=[]  
                    c.execute('SELECT time,name,countt FROM drinks')
                    read_list1=c.fetchall()
                    df1= pd.DataFrame(
                        read_list1,
                        columns=["time","name","count"]
                    )
                    st.table(df1) 

                    read_list2=[]    
                    c.execute('SELECT name,sum(countt) FROM drinks GROUP BY name ')
                    read_list2=c.fetchall()
                    df2 = pd.DataFrame(
                        read_list2,
                        columns=["name","count"]
                    )
                    st.table(df2)  


                elif(choice=='Biscuits'):

                    biscuitname=st.text_input("Enter biscuit name")
                    biscuitcount=st.text_input("Enter count")
                    date=st.date_input("Select date")
                    

                    if(st.button("SUBMIT")):
                            c.execute('INSERT INTO biscuits (time,name,countt) VALUES (?, ?, ?)',( date, biscuitname , biscuitcount))
                            conn.commit()
                    
                    read_list1=[]  
                    c.execute('SELECT time,name,countt FROM biscuits')
                    read_list1=c.fetchall()
                    df1= pd.DataFrame(
                        read_list1,
                        columns=["time","name","count"]
                    )
                    st.table(df1) 

                    read_list2=[]    
                    c.execute('SELECT name,sum(countt) FROM biscuits GROUP BY name ')
                    read_list2=c.fetchall()
                    df2 = pd.DataFrame(
                        read_list2,
                        columns=["name","count"]
                    )
                    st.table(df2)  



                elif(choice=='Lays'):

                    laysname=st.text_input("Enter lays name")
                    layscount=st.text_input("Enter count")
                    date=st.date_input("Select date")
                    

                    if(st.button("SUBMIT")):
                            c.execute('INSERT INTO lays (time,name,countt) VALUES (?, ?, ?)',( date, laysname , layscount))
                            conn.commit()
                    
                    read_list1=[]  
                    c.execute('SELECT time,name,countt FROM lays')
                    read_list1=c.fetchall()
                    df1= pd.DataFrame(
                        read_list1,
                        columns=["time","name","count"]
                    )
                    st.table(df1) 

                    read_list2=[]    
                    c.execute('SELECT name,sum(countt) FROM lays GROUP BY name ')
                    read_list2=c.fetchall()
                    df2 = pd.DataFrame(
                        read_list2,
                        columns=["name","count"]
                    )
                    st.table(df2)  


                elif(choice=='Shampoos'):

                    shampooname=st.text_input("Enter shampoo name")
                    shampoocount=st.text_input("Enter count")
                    date=st.date_input("Select date")
                    

                    if(st.button("SUBMIT")):
                            c.execute('INSERT INTO shampoos (time,name,countt) VALUES (?, ?, ?)',( date, shampooname , shampoocount))
                            conn.commit()
                    
                    read_list1=[]  
                    c.execute('SELECT time,name,countt FROM shampoos')
                    read_list1=c.fetchall()
                    df1= pd.DataFrame(
                        read_list1,
                        columns=["time","name","count"]
                    )
                    st.table(df1) 

                    read_list2=[]    
                    c.execute('SELECT name,sum(countt) FROM shampoos GROUP BY name ')
                    read_list2=c.fetchall()
                    df2 = pd.DataFrame(
                        read_list2,
                        columns=["name","count"]
                    )
                    st.table(df2)  
                        

                else:
                    if(choice=='Noodles'):

                        noodlesname=st.text_input("Enter noodles name")
                        noodlescount=st.text_input("Enter count")
                        date=st.date_input("Select date")
                        

                        if(st.button("SUBMIT")):
                                c.execute('INSERT INTO noodles (time,name,countt) VALUES (?, ?, ?)',( date, noodlesname , noodlescount))
                                conn.commit()
                        
                        read_list1=[]  
                        c.execute('SELECT time,name,countt FROM noodles')
                        read_list1=c.fetchall()
                        df1= pd.DataFrame(
                            read_list1,
                            columns=["time","name","count"]
                        )
                        st.table(df1) 

                        read_list2=[]    
                        c.execute('SELECT name,sum(countt) FROM noodles GROUP BY name ')
                        read_list2=c.fetchall()
                        df2 = pd.DataFrame(
                            read_list2,
                            columns=["name","count"]
                        )
                        st.table(df2)  

                choice5=st.selectbox("Download Mode",["None", "CSV","Excel"])

                if choice5=="CSV":
                    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
                    st.download_button(label='download csv',data=df.to_csv() ,mime='text/csv' ,)

                elif choice5=="Excel":
                    st.download_button(label='download excel',data="abc.xlsx" ,mime='text/xlsx')
                    
            if selected=="Video":
            
                        st.title('📹Object detection video')
                        date1=st.date_input("Date")
                        confidence_threshold=st.slider("Confidence threshold",0.0,1.0)

                        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                        
                        webrtc_ctx = webrtc_streamer(
                            key="webcam",
                            mode=WebRtcMode.SENDRECV,
                            rtc_configuration=RTC_CONFIGURATION,
                            media_stream_constraints={"video": True, "audio": False},
                            video_processor_factory=VideoProcessor,
                            async_processing=True,
                        )
                        if webrtc_ctx.state.playing:
                                webrtc_ctx.video_processor.confidence=confidence_threshold
                        if st.checkbox('Show the table'):
                            empty=st.empty()
                            store=st.button('Store')
                            if webrtc_ctx.state.playing:
                                while True:
                                    if webrtc_ctx.video_processor:
                                        result = webrtc_ctx.video_processor.getRes()
                                        if result!= None:
                                            count = result.pandas().xyxy[0]['name'].value_counts()
                                            empty.write(count)
                                            
                                                
                                        else:
                                            empty.write("No labels detected")  
                                    else:
                                        break
main()
                    
                

         
#if LOGGED_IN == True:
#    st.markdown("Welcome to Main page!")


    # Get the user name.
fetched_cookies = __login__obj.cookies
if '_streamlit_login_signup_ui_username_' in fetched_cookies.keys():
    username = fetched_cookies['_streamlit_login_signup_ui_username_']
    st.markdown(st.session_state)
    st.write(username)

#if LOGGED_IN == True:

#   st.markdown("Your Streamlit Application Begins here!")
 #  st.markdown(st.session_state)'''
   #st.write(username)


#from st_login_form import login_form
#client = login_form()
                                      