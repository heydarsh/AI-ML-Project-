# Mushroom Analysis and Classification Web Application

## Overview

This Django-based web application enables users to upload mushroom datasets, analyze their characteristics, and predict edibility using machine learning. The platform provides interactive visualizations, statistical insights, and downloadable results, making it a valuable tool for researchers, students, and mushroom enthusiasts.

---

## Features

- **CSV Dataset Upload**: Upload mushroom datasets for analysis.
- **Automated Data Validation**: Ensures required columns and data types are present.
- **Interactive Visualizations**: Distribution plots, correlation heatmaps, feature importance, and more.
- **Machine Learning Classification**: Logistic Regression-based prediction of mushroom edibility.
- **Performance Metrics**: Accuracy, classification report, and confusion matrix.
- **Downloadable Results**: Export analysis in CSV or JSON format.
- **Robust Error Handling**: User-friendly error messages and logging.
- **Responsive Web Interface**: Built with Bootstrap for usability.

---

## Technology Stack

- **Backend**: Django 4.2+, Python 3.8+
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Database**: SQLite3 (default), PostgreSQL (optional)
- **Testing**: Django TestCase, pytest

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://https://github.com/heydarsh/AI-ML-Project
    cd mushroom-analysis
    ```

2. **Create and activate a virtual environment**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Unix/Mac:
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Apply migrations**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

5. **Run the development server**
    ```bash
    python manage.py runserver
    ```

6. **Access the application**
    - Open your browser and go to: [http://localhost:8000](http://localhost:8000)

---

## Usage

1. Upload a CSV file containing mushroom data.
2. Select the desired machine learning algorithm (default: Logistic Regression).
3. View interactive charts and model performance metrics.

**Sample CSV columns:**
- `class` (edible/poisonous)
- `cap-diameter`
- `stem-height`
- `stem-width`
- `cap-shape`
- `cap-color`
- `gill-color`

---

## Project Structure

```
project/
├── chartapp/
│   ├── views.py
│   ├── models.py
│   ├── forms.py
│   ├── utils/
│   └── templates/
├── static/
├── requirements.txt
└── manage.py
```

---

## Error Handling

- Validates file type and size on upload.
- Checks for required columns and data types.
- Handles missing or malformed data gracefully.
- Displays user-friendly error messages in the UI.
- Logs errors for debugging.

---

## Results and Outputs

- **Accuracy** and **classification report** for model predictions.
- **Confusion matrix** and feature importance charts.
- Downloadable analysis results in CSV or JSON format.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Conclusion

The Mushroom Analysis Application streamlines the process of analyzing mushroom datasets and predicting edibility. With its robust data validation, interactive visualizations, and machine learning integration, it serves as a practical tool for research, education, and exploration in mycology and data science.

---
