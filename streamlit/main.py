import helper
import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as components

image = Image.open('/Users/ashwin.gangadhar/projects/mdb-bfsi-genai/image/credit-card-image.jpg')

st.title("BFSI: User Profiling and Credit Card Suggestion")

with st.sidebar:
    # with st.expander("By Age"):
    #     customer_age = st.selectbox("Pick a customer profile age gt than", ("25", "35", "50", "65"))
    #     st.checkbox("Defaulter", key="defaulter")

    cust_id = st.text_input('Customer ID')
    get_cust_id = st.button("Get", type="primary")

    # if st.session_state.defaulter:
    #     dlq_status = True
    # else:
    #     dlq_status = False

    flag = False

    # if customer_age:
    #     feature_importance, features, dlq = helper.get_user_n_model_info(customer_age,dlq_status=dlq_status)
    #     flag=True
        

    if cust_id and get_cust_id:
        feature_importance, features, dlq = helper.get_user_n_model_info_by_id(int(cust_id))
        flag=True

    if flag:
        st.title("*User profile features*")
        for k,v in features.items():
                st.metric(label="**"+k+"**", value=v)

b = st.sidebar.button("Process", type="primary")

if b:
    feature_importance, features, dlq = helper.get_user_n_model_info_by_id(int(cust_id))    
    response = helper.generate_flow(feature_importance, features, dlq)
    tab1, tab2 = st.tabs(["User profile summary", "Recommendations"])
    with tab1:
        st. markdown('<p style=“font-size:50;”>'+response["response"]+'</p>', unsafe_allow_html=True) 
        # st.write(response["response"])
    with tab2:
        if dlq<0.4:
            reco_items = response['out']
            st.title("**Recommendations**")
            for i,item in enumerate(reco_items):
                if i==0:
                    continue
                if len(item.split(":"))>1:
                    with st.container():
                        col1, col2 = st.columns(2)
                        cname, desc = item.split(":")
                        with col1:
                            # txt = """<img alt="Marriott Bonvoy HDFC Bank Credit Card - Hotel Credit Card" src="/content/api/contentstream-id/723fb80a-2dde-42a3-9793-7ae1be57c87f/4b571498-9b07-4384-ae28-1fd1c55c3d0f/Personal/Pay/Cards/Credit Card/Credit Card Landing Page/Credit Cards/Co-Brand/Marriott Co-Brand Credit card/Marriotte-Bonvoy-Brand-Credit-Card-264x167.png" title="Marriott Bonvoy HDFC Bank Credit Card - Hotel Credit Card">"""
                            st.image(image)
                            st.header(cname.split(".")[-1].strip())
                            # st. markdown('<p style=“font-size:100;”>'+cname.split(".")[-1].strip()+'</p>', unsafe_allow_html=True)
                            # st.header("**"+cname.split(".")[-1].strip()+"**")
                            # components.html(txt)
                            # st.image<img src="img_girl.jpg" alt="Girl in a jacket" width="500" height="600">
                        with col2:
                            st.caption(desc)
                            #  st. markdown('<p style=“font-size:24;”>'+desc+'</p>', unsafe_allow_html=True)
