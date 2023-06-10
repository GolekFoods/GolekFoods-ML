# GolekFoods Machine Learning Design

## About GolekFoods Machine Learning Design
The Machine Learning design of the **GolekFoods** website is intended to provide food recommendations according to user preferences. We hope that by utilizing machine learning, the process will run faster and more accurately considering the amount of data we have is quite large and it is almost impossible to make logic with conventional programming techniques. Furthermore, we put the finished machine learning model into the Flask framework to be made into an API that can be consumed by the **GolekFoods** website.

Rancangan Machine Learning dari website **GolekFoods** ini dimakhsudkan untuk memberikan rekomendasi makanan sesuai preferensi pengguna. Kami berharap dengan memanfaatkan machine learning maka proses akan berjalan lebih cepat dan akurat mengingat jumlah data yang kami miliki cukup banyak dan hampir mustahil untuk membuat logika dengan teknik pemrograman konvensional. Selanjutnya model machine learning yang sudah jadi kami masukkan ke dalam framework Flask untuk kemudian dibuat menjadi API yang dapat dikonsumsi oleh website **GolekFoods**.

## Resources

### Tools
- Visual Studio Code

### Programming Language
- Python

### Library and Framework
- Pandas
- Numpy
- Scikit-Learn
- Joblib
- Flask-cors
- Flask
- Gunicorn

### API
- [Deployed GolekFoods Machine Learning API](https://golekfoodsflask-production.up.railway.app/)

### Algorithm
- Support Vector Machine
- K-Nearest Neighbors
- Random Forest
- Gaussian Naive Bayes **(Deployed for production phase)**

### Dataset 
- [Data Komposisi Pangan Indonesia](https://www.panganku.org/) (Web Scraping, we add the image column manually)

## Documentation
1. Clone this repository

```
git clone https://github.com/GolekFoods/GolekFoods-ML.git
```

2. Install all requirements.txt

```
pip install -r requirements.txt
```

3. Run locally with Flask

```
python -m flask run
```