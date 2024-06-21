import pickle
import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

st.header("Sistem Rekomendasi Buku dengan ML")

# Memuat model dan data
model = pickle.load(open('model/model.pkl', 'rb'))
dataset = pickle.load(open('model/dataset.pkl', 'rb'))
books_name = pickle.load(open('model/books_name.pkl', 'rb'))
final_rating = pickle.load(open('model/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('model/book_pivot.pkl', 'rb'))
age_recommend_dataset = pickle.load(open('model/age_recommend_dataset.pkl', 'rb'))

books_rate_df = dataset.groupby('Book-Title')

# Fungsi untuk mendapatkan buku populer
def populer_books():
    return books_rate_df.size().sort_values(ascending=False).head(10).index

# Fungsi untuk mendapatkan buku populer berdasarkan Rating Tertinggi
def highest_rated_books():
    return books_rate_df['Book-Rating'].mean().sort_values(ascending=False).head(10).index

# Fungsi untuk merekomendasikan buku berdasarkan umur
def age_based_recommendation(age, group):
    age_data = age_recommend_dataset[age_recommend_dataset['Age-Group'] == group]
    age_data = age_data.reset_index()
    result = []
    for i in range(age_data.shape[0]):
        # Kolom 2 adalah kolom umur
        diff = abs(age_data.iloc[i, 2] - age)
        result.append([age_data.iloc[i, 0], diff])
    result.sort(key=lambda x: x[1])
    recommended_books = [result[i][0] for i in range(min(10, len(result)))]
    return recommended_books

# Menampilkan buku populer
st.subheader("Buku Populer")
populer_books_list = populer_books()
populer_books_df = pd.DataFrame(populer_books_list, columns=['Book-Title'])
st.write("------- Berikut adalah beberapa buku populer yang mungkin Anda suka -------")
st.write("-----Selamat Membaca!!!-------")
st.dataframe(populer_books_df)

# Filtering books on highest average ratings
st.subheader("Buku Populer dengan Rating Tertinggi")
highest_rated_list = highest_rated_books()
highest_rated_df = pd.DataFrame(highest_rated_list, columns=['Book-Title'])
st.write("------- Berikut adalah beberapa buku dengan rating tertinggi yang mungkin Anda suka -------")
st.write("-----Selamat Membaca!!!-------")
st.dataframe(highest_rated_df)

st.subheader("Berdasarkan Umur")

# Input untuk umur pengguna
age = st.number_input('Masukkan umur Anda', min_value=1, max_value=100, value=26)

# Menampilkan rekomendasi berdasarkan umur
if age <= 30:
    age_group = 'young'
elif 30 < age <= 60:
    age_group = 'mid-age'
else:
    age_group = 'old'

age_based_recommendations = age_based_recommendation(age, age_group)
st.subheader("Rekomendasi Buku Berdasarkan Umur")
if age_based_recommendations:
    age_recommend_df = dataset.loc[age_based_recommendations][['Book-Title']].drop_duplicates().reset_index()
    st.write(age_recommend_df['Book-Title'])
else:
    st.write("Tidak ada rekomendasi berdasarkan umur yang tersedia.")

# Fungsi untuk menghitung kesamaan Euclidean
def euclidSim(inA, inB):
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))

# Fungsi untuk menghitung kesamaan Pearson
def pearsonSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]

# Fungsi untuk memperkirakan rating menggunakan SVD
def svdEst(dataMat, user, simMeas, item, n_components=4):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    svd = TruncatedSVD(n_components=n_components)
    xformedItems = svd.fit_transform(dataMat.T)
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :], xformedItems[j, :])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

# Fungsi untuk merekomendasikan buku menggunakan SVD
def recommend(dataMat, user, N=3, simMeas=pearsonSim, estMethod=svdEst, n_components=4):
    unratedItems = np.nonzero(dataMat[user, :] == 0)[0]  # Menemukan item yang belum diberi rating
    if len(unratedItems) == 0:
        return 'Anda sudah memberi rating semua item'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item, n_components=n_components)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

# Fungsi untuk merekomendasikan buku berdasarkan model KNN
def rekomendasi_buku(selected_book_name):
    book_list = []
    if selected_book_name in book_pivot.index:
        book_id = np.where(book_pivot.index == selected_book_name)[0]
        if len(book_id) == 0:
            st.error(f"Buku '{selected_book_name}' tidak ditemukan dalam indeks.")
            return book_list
        
        book_id = book_id[0]
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=10)

        for i in range(len(suggestion[0])):
            book_id_suggested = suggestion[0][i]
            book_list.append(book_pivot.index[book_id_suggested])
    else:
        st.error(f"Buku '{selected_book_name}' tidak ditemukan dalam indeks.")
    
    return book_list

# Dropdown untuk memilih buku
books_name_list = list(book_pivot.index)
selected_books = st.selectbox("Ketik atau pilih sebuah buku", books_name_list)

# Tombol untuk menampilkan rekomendasi
if st.button('Tampilkan Rekomendasi'):
    if selected_books:
        recommended_books = rekomendasi_buku(selected_books)
        
        # Menampilkan rekomendasi secara vertikal
        if recommended_books:
            st.write("Rekomendasi Buku:")
            for book in recommended_books:
                st.write(book)
    else:
        st.error("Silakan pilih sebuah buku terlebih dahulu.")

# Tombol untuk menampilkan rekomendasi menggunakan SVD
if st.button('Tampilkan Rekomendasi Khusus'):
    with st.spinner('Sedang memproses rekomendasi...'):
        start_time = time.time()
        user = 18  # Contoh user ID
        recommendations = recommend(book_pivot.values, user, N=5, simMeas=pearsonSim, estMethod=svdEst, n_components=4)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        st.write(f"Waktu yang diperlukan untuk rekomendasi SVD: {elapsed_time:.2f detik}")
        
        # Menampilkan rekomendasi untuk user tertentu
        st.write(f"Item yang Direkomendasikan untuk User {user}:\n")
        for i, score in recommendations:
            st.write(f'Buku: "{book_pivot.index[i]}" , Rating : "{round(score, 2)}"')
