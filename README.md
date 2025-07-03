# 🧠 Visual Search Engine using Deep Learning and FAISS

A content-based image retrieval prototype that uses a custom-trained Convolutional Neural Network (CNN) on CIFAR-10 and a FAISS vector database to compute and retrieve image embeddings. Built with PyTorch and deployed via Streamlit.

![Visual Search Engine](https://github.com/user-attachments/assets/ad24f953-3d52-483c-9f77-cee7c82adfc1)

---

## 💡 Motivation & Use Case

Reverse image search is increasingly important in e-commerce, medical imaging, and multimedia organization. This project demonstrates how to build a compact yet powerful visual recognition engine that predicts image classes and is capable of retrieving similar images via deep learning embeddings and FAISS indexing.

Real-world use cases include:

* 📷 Visual search in shopping apps (e.g., "Find similar products")
* 🧬 Medical imaging (e.g., "Find similar X-rays or skin conditions")
* 🎨 Organizing large-scale image databases

---

## 🚀 Features

* 🧠 CNN trained from scratch on CIFAR-10 to extract meaningful image embeddings
* 🔍 Predicts the top class of an uploaded image using deep features
* ⚡ Fast vector similarity search using FAISS for retrieving similar images
* 🖼️ Streamlit-based interactive UI

---

## 🛠️ Tech Stack

* **PyTorch** – Model training and inference
* **FAISS** – Efficient vector similarity search
* **Torchvision** – CIFAR-10 dataset and transforms
* **Streamlit** – Web-based frontend for user interaction
* **Pillow** – Image handling

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/akhil-b-26/visual-search-dl-faiss.git
cd visual-search-dl-faiss
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🧪 Run the App

```bash
streamlit run app.py
```

Then open your browser to [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deployment (Streamlit Cloud)

You can deploy this app for free on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push your code to a public GitHub repository
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub and click **“New App”**
4. Select the repository and `app.py` as the entrypoint
5. Click **“Deploy”**

Your app will be live on a link like:

```
https://your-username.streamlit.app
```

---

## 🖼️ Example Usage

> ⚠️ Note: The current model achieves 77.18% accuracy on CIFAR-10. Due to the dataset’s small size and low resolution, occasional misclassifications (e.g., fish labeled as airplane) may occur. Improving training duration or using a deeper model (e.g., ResNet18) can enhance accuracy.

* Upload a 32×32 image (or any image, it will be resized)
* The app predicts the top class of the image using the trained CNN
* 🔍 Retrieves and displays the top 5 visually similar images using FAISS and deep embeddings
  
---

## 📂 Folder Structure

```
visual-search-dl-faiss/
├── app.py
├── src/
│   ├── model_training.ipynb
│   ├── cnn_model.pth
│   ├── embeddings.npy
│   ├── indices.npy
│   ├── classes.pkl
├── requirements.txt
└── README.md

```

---

## 🌟 License

This project is open source and free to use under the [MIT License](LICENSE).
