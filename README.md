# Crop Recommendation ML Project

This machine learning project aims to recommend the most suitable crop to grow based on various soil and environmental factors. The model takes into account the following input data:

- **N**: Ratio of Nitrogen content in the soil
- **P**: Ratio of Phosphorous content in the soil
- **K**: Ratio of Potassium content in the soil
- **Temperature**: Temperature in degree Celsius
- **Humidity**: Relative humidity in %
- **pH**: pH value of the soil
- **Rainfall**: Rainfall in mm

## Deployment

Url: https://subhan97ahmed-croprecommendation-frontendindex-ffi2go.streamlit.app/

## Dataset

The dataset used for training and testing the crop recommendation model consists of historical records of soil and environmental factors along with the corresponding crops that were grown. The dataset is in a structured format, typically in a CSV (Comma Separated Values) file.

## Model Training

The crop recommendation model is built using machine learning algorithms. The training process involves feeding the model with the labeled dataset, allowing it to learn the patterns and relationships between the input features (N, P, K, Temperature, Humidity, pH, Rainfall) and the target variable (Crop).

The training process may include various steps, such as data preprocessing (cleaning, scaling, and feature engineering), model selection, hyperparameter tuning, and evaluation. The specific machine learning algorithm used can vary, ranging from decision trees, random forests, support vector machines (SVM), to deep learning models like neural networks.

## Streamlit Frontend

The Streamlit frontend provides an interactive web interface to access the crop recommendation model. It allows users to input the soil and environmental factors and obtain the recommended crop as the output.

To run the Streamlit app, follow these steps:

1. Install the required dependencies. You can find the necessary packages and their versions in the `requirements.txt` file.
```
pip install -r requirements.txt
```
2. Open a terminal or command prompt, navigate to the directory where the `index.py` file is saved, and run the following command:

```
streamlit run index.py
```

3. The Streamlit app will start a local server and open the app in your default web browser. You will see a web interface with sliders for each input field. Adjust the sliders to set the desired values for the soil and environmental factors.

4. The recommended crop will be displayed below the sliders based on the input values.

## Acknowledgements

The crop recommendation model may have been developed using publicly available datasets or datasets specific to a particular region. Acknowledgments go to the original data sources and any additional contributors who have provided the necessary information for developing the model.

## License

The crop recommendation project is released under the [MIT License](LICENSE). You are free to modify, distribute, and use the code as per the terms and conditions of the license.

## Conclusion

The crop recommendation ML project offers a valuable tool for farmers and agricultural practitioners to make informed decisions about which crops to grow based on soil and environmental factors. By leveraging machine learning algorithms and providing a user-friendly interface through Streamlit, the model provides personalized recommendations tailored to the specific conditions of a given area. Feel free to explore and experiment with the project to enhance its accuracy and usability.
