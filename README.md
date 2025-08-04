# 🎧 Environmental Sound Classification using ANN (Multilayer Perceptron)

This project is an **audio classification system** built with an **Artificial Neural Network (ANN)** (Multilayer Perceptron) that classifies environmental sounds such as sirens, dog barks, children playing, and more. It uses **MFCC (Mel Frequency Cepstral Coefficients)** as input features extracted from the **UrbanSound8K** dataset.

---

## 📌 Highlights

- 🔊 Classifies 10 types of urban environmental sounds
- 🎼 MFCC-based feature extraction using `librosa`
- 🧠 ANN-based classifier built with TensorFlow/Keras
- ✅ Achieves **~93% accuracy** on test data
- 📈 Confusion matrix & classification report for evaluation
- 🌐 Real-time prediction using Streamlit

---

## 🔊 Dataset: UrbanSound8K

- 8732 labeled audio clips (≤4 seconds)
- 10 categories of urban sounds:

| Label | Class             |
|-------|-------------------|
| 0     | Air Conditioner   |
| 1     | Car Horn          |
| 2     | Children Playing  |
| 3     | Dog Bark          |
| 4     | Drilling          |
| 5     | Engine Idling     |
| 6     | Gun Shot          |
| 7     | Jackhammer        |
| 8     | Siren             |
| 9     | Street Music      |

Download from [UrbanSound8K Website](https://urbansounddataset.weebly.com/urbansound8k.html)

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/audio-ann-classification.git
cd audio-ann-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

- Download UrbanSound8K
- Extract it inside the `data/` folder (i.e., `data/UrbanSound8K/`)

### 4. Extract MFCC features

```bash
python preprocessing/extract_features.py
```

### 5. Train the ANN model

```bash
python train_model.py
```

### 6. Evaluate the model

```bash
python evaluate_model.py
```

### 7. Launch Streamlit App (optional)

```bash
streamlit run streamlit_app.py
```

---

## 📊 Model Performance

| Metric     | Value      |
|------------|------------|
| Accuracy   | 93.02%     |
| Test Loss  | 0.3165     |
| Model Type | ANN (Dense layers with Dropout + Softmax) |

**Classification Report Example:**

```
              precision    recall  f1-score   support

        Siren       0.86      0.88      0.87       200
     Dog Bark       0.98      0.97      0.98       216
     Children       0.90      0.90      0.90       182
        ...
```

**Confusion Matrix Heatmap:**

> Automatically generated using Seaborn in `evaluate_model.py`

---

## 🧪 Technologies Used

- **Python 3.11**
- **TensorFlow/Keras** – Model building & training
- **Librosa** – Audio processing & MFCC extraction
- **NumPy / Pandas** – Data manipulation
- **Matplotlib / Seaborn** – Plotting
- **Streamlit** – For real-time web-based predictions

---

## 🌐 Streamlit App Features

- Upload `.wav` file
- Audio gets preprocessed and converted to MFCC
- Model predicts the sound category
- Label is displayed in real time

---


## ✍️ Author

**Nitin Pratap**  
[GitHub](https://github.com/Nitinpratap22061/)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and share.

---

## 🙏 Acknowledgments

- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
- [Librosa Audio Library](https://librosa.org/)
- [Keras/TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
