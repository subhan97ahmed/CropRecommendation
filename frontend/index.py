import pickle
import pandas as pd
import streamlit as st
import os


def main():
    st.title("Crop Recommendation")

    # Text input fields
    N = st.text_input("Ratio of Nitrogen content in soil")
    P = st.text_input("Ratio of Phosphorous content in soil")
    K = st.text_input("Ratio of Potassium content in soil")
    temperature = st.text_input("Temperature in degree Celsius")
    humidity = st.text_input("Relative humidity in %")
    ph = st.text_input("pH value of the soil")
    rainfall = st.text_input("Rainfall in mm")

    result = st.button("Predict")
    if result:
        if N == "" or P == "" or K == "" or temperature == "" or humidity == "" or ph == "" or rainfall == "":
            st.error("Fill All The Fields")
        else:
            data = {
                "N": [N],
                "P": [P],
                "K": [K],
                "temperature": [temperature],
                "humidity": [humidity],
                "ph": [ph],
                "rainfall": [rainfall],
            }
            data = pd.DataFrame(data)

            print(data.head())
            print(os.path)
            model = pickle.load(open('model/model/RandomForestClassifier.sav', 'rb'))
            pred = model.predict(data)
            st.success(str(pred[0]), icon="✅")
            print(pred)


if __name__ == '__main__':
    main()
