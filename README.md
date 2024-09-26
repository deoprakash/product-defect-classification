
---

# Product Defect Classification using Deep Learning

This repository contains a **Product Defect Classification** project that uses a deep learning model to detect whether a product is defective or not. The project is built with **TensorFlow** using the **MobileNetV2** pre-trained model for image classification and features a simple web interface for users to upload product images and receive predictions.

### Features:
- **Deep Learning Model**: Utilizes **MobileNetV2** for efficient image classification with transfer learning.
- **Image Upload and Prediction**: Allows users to upload images of products and receive a prediction on whether the product is defective or non-defective.
- **Web Interface**: The project includes a web-based interface built with **Flask**, where users can upload images, view predictions, and see detailed descriptions.
- **Prediction Table**: Displays a list of uploaded images along with their predicted class and description.
- **Temporary Storage**: Images are temporarily stored during the session for operations without being saved on the local machine.
- **Clear Data**: The prediction data is cleared when the browser is refreshed, ensuring no lingering data.

### Technologies:
- **TensorFlow**: Deep learning framework for building and training the defect classification model.
- **Flask**: A lightweight web framework used for creating the web interface.
- **HTML/CSS**: For building the front-end web interface.
- **Local Storage**: For handling temporary image storage during the session.

### Installation:
1. Clone the repository:
   ```
   git clone https://github.com/your-username/product-defect-classification.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```
   python app.py
   ```
4. Open your browser and visit `http://localhost:5000` to start using the application.

### Usage:
- Upload an image of a product.
- View the prediction result and detailed description in a table format.
- The data clears when the browser is refreshed, ensuring privacy and security.

### Future Improvements:
- Enhance the accuracy of the model with more training data.
- Implement a database to store historical predictions.
- Add user authentication and role-based access control.

### Contributing:
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.

---

