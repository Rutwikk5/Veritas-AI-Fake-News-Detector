# 🕵️‍♀️ Veritas AI — Multimodal Fake News Detection

A **multimodal fake news detection system** that analyzes **textual headlines** and **associated images** to classify news as **Real** or **Fake**.
The system combines predictions from a **text model (BERT)** and an **image model (ResNet50)** using **late fusion** with configurable weights.

---

## 📌 Project Overview

Fake news often exploits **both misleading text and deceptive images**. Single-modal systems fail when one modality looks legitimate.
This project addresses that gap by:

* Analyzing **headline text** using a transformer-based NLP model
* Analyzing **news images** using a CNN-based vision model
* Combining both predictions using **late fusion**
* Allowing **dynamic control of modality importance** via UI

---

## 🧠 Architecture

**Text Modality**

* Model: BERT (Fine-tuned)
* Input: News headline
* Output: Probability of fake news

**Image Modality**

* Model: ResNet50 (Fine-tuned)
* Input: News image
* Output: Probability of fake news

**Fusion Strategy**

* Late Fusion (Weighted Average)
* Final score =
  `w_text × P_text + w_image × P_image`
* Weights always sum to 1.0 and are adjustable in real time

---

## 🖥️ Application Interface

The system is deployed as a **Streamlit web application** with:

* Text input for news headline
* Image upload for related news image
* Slider to control text vs image importance
* Clear verdict display: **REAL** or **FAKE**
* Confidence score and per-modality breakdown

---

## 📂 Project Structure

```
├── app.py                      # Streamlit application
├── final_main.ipynb            # Training & evaluation notebook
├── final_text_model.h5         # Trained BERT text model
├── final_image_model.keras     # Trained ResNet50 image model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/veritas-ai.git
cd veritas-ai
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🔬 Model Details

### Text Model

* Tokenizer: `bert-base-uncased`
* Input length: 64 tokens
* Output: Fake probability (binary classification)

### Image Model

* Input size: 160 × 160 RGB
* Preprocessing: ResNet50 preprocessing
* Output: Fake probability (binary classification)

---

## 📊 Output Explanation

* **Final Verdict**: REAL or FAKE
* **Confidence Score**: Probability of predicted class
* **BERT Score**: Fake probability from text
* **ResNet Score**: Fake probability from image

This breakdown ensures **interpretability**, not just blind prediction.

---

## 🚫 Limitations

Let’s be honest:

* No cross-modal attention (modalities are processed independently)
* Performance depends heavily on dataset quality
* Only headline text is used, not full articles
* Fusion weights are manually controlled, not learned

All of this is intentional and clearly scoped.

---

## 🚀 Future Improvements

* Learned fusion instead of manual weights
* Cross-modal attention mechanisms
* Full article text analysis
* Explainability using Grad-CAM and attention visualization
* Real-time news scraping integration

---

## 📜 License

This project is intended for **academic and educational use**.

---
