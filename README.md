# 🎓 NLP-Based Course Recommendation System

An intelligent course recommendation system that suggests relevant online courses based on a user's interests using **Natural Language Processing (NLP)** and **Sentence Transformers**. The system understands the semantic meaning of user queries and recommends the most relevant courses from a curated dataset.

---
## 🚀 Live Demo

Experience the Course Recommendation System online without any installation.

🔗 **Live Application:**  
https://nlp-based-course-recommended-system-chnafdjxfiocxgpf4e6nq5.streamlit.app/

## 📌 Features

* 🔍 Semantic course search using natural language queries.
* 🤖 NLP-powered recommendations using Sentence Transformers.
* ⭐ Ranks courses based on semantic similarity and course ratings.
* 📚 Displays top recommended courses and additional related courses.
* 🌐 Interactive web application built with Streamlit.
* 🎨 Modern and user-friendly interface.

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* Sentence Transformers
* Scikit-learn
* Pickle

---

## 📂 Project Structure

```text
Course-Recommendation-System/
│── app.py
│── NLP_based_course_recommender.ipynb
│── final_courses.csv
│── course_embeddings.pkl
│── requirements.txt
│── README.md
└── .gitignore
```

---

## ⚙️ How It Works

1. The user enters a course topic or skill (e.g., Machine Learning, Python, Data Science).
2. The input text is converted into a semantic embedding using the **all-MiniLM-L6-v2** Sentence Transformer model.
3. The query embedding is compared with precomputed course embeddings using **Cosine Similarity**.
4. Courses with high similarity scores are filtered and ranked.
5. The final ranking combines:

   * **70% Semantic Similarity**
   * **30% Course Rating**
6. The system displays the top recommendations along with additional related courses.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/SATYA012904/NLP-based-course-recommended-system.git
```

Navigate to the project directory:

```bash
cd NLP-based-course-recommended-system
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📊 Dataset

The project uses a curated dataset of online courses containing information such as:

* Course Title
* Instructor
* Platform
* Rating
* Course URL

The dataset is processed to generate semantic embeddings for efficient course retrieval.

---

## 🧠 NLP Pipeline

* Text Preprocessing
* Sentence Embedding Generation
* Cosine Similarity Calculation
* Similarity Threshold Filtering
* Duplicate Removal
* Rating Normalization
* Final Score Ranking



```
assets/
├── home.png
└── recommendations.png
```

---

## 🔮 Future Enhancements

* Personalized recommendations based on user history.
* Multi-language course search.
* Course bookmarking and favorites.
* Integration with additional learning platforms.
* User authentication and profiles.

---

## 👨‍💻 Author

**Satyabrata Sahu**

B.Tech (Artificial Intelligence & Machine Learning)

---

## 📄 License

This project is developed for educational and learning purposes.
