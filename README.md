# Multi-Disease Prediction System

This project is a **Multi-Disease Prediction System** that predicts the likelihood of multiple diseases—**Liver Disease**, **Kidney Disease**, and **Parkinson's Disease**—based on user input. The system uses **Streamlit** to create a user-friendly interface for real-time predictions.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- - [Contact](#contact)

## Features
- Predicts three diseases: Liver, Kidney, and Parkinson's.
- Interactive and intuitive web-based interface built with Streamlit.
- Leverages machine learning models for accurate predictions.
- Outputs the results based on user-provided medical data.

## Technologies Used
- **Frontend**: [Streamlit](https://streamlit.io/) for the GUI.
- **Backend**: Python for data processing and machine learning.
- **Libraries**:
  - `pandas` for handling datasets.
  - `scikit-learn` for model training and predictions.
  - `numpy` for numerical computations.
  - `streamlit` for building the frontend interface.
  - `matplotlib` and `seaborn` for data visualization (optional).

## Datasets
The project is powered by three datasets:
1. **Liver Disease Dataset**: Includes patient data to predict liver-related health issues.
2. **Kidney Disease Dataset**: Contains features and labels for kidney disease diagnosis.
3. **Parkinson's Disease Dataset**: Comprises symptoms and diagnostic indicators for Parkinson's disease.

> **Note**: Ensure datasets are placed in the appropriate directory before running the application.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Keerthana2413/multi-disease-prediction.git
   cd multi-disease-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the URL provided by Streamlit (usually `http://localhost:8501`) in your browser.
2. Select the disease you want to predict: Liver, Kidney, or Parkinson's.
3. Enter the required patient details in the form.
4. Click the **Predict** button to view the results.

## Contributing
Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or suggestions, feel free to reach out:
- **GitHub**: [Keerthana2413](https://github.com/Keerthana2413)
- **Email**: keerthishine139@gmail.com
