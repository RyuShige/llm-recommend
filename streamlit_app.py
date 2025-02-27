import numpy as np
import faiss
import streamlit as st
import zipfile
import requests
import random
from openai import OpenAI
import re

# OpenAIとOMDb APIのクライアント設定
client = OpenAI()
client.api_key = st.secrets["OPENAI_API_KEY"]
omdb_api_key = st.secrets["OMDB_API_KEY"]

# 映画データの読み込み
zip_file_path = 'movie_info.zip'
with zipfile.ZipFile(zip_file_path, 'r') as z:
    file_list = sorted(z.namelist())
    with z.open(file_list[0]) as id2emb_file:
        id2emb = np.load(id2emb_file, allow_pickle=True)
    with z.open(file_list[1]) as movie_description_file:
        movie_description = np.load(movie_description_file, allow_pickle=True)
    with z.open(file_list[2]) as movie_title_file:
        movie_title = np.load(movie_title_file, allow_pickle=True)

# FAISSインデックスの読み込み
index = faiss.read_index('movie_index.faiss')

# 映画情報を取得する関数
def get_movie_info(title):
    params = {'apikey': omdb_api_key, 't': title}
    response = requests.get('http://www.omdbapi.com/', params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get('Response') == 'True':
            return {
                'title': data.get('Title'),
                'poster_url': data.get('Poster'),
                'imdb_id': data.get('imdbID')
            }
    return None

# ランダムに映画を選択する関数
def get_random_movies(num_movies=8):
    total_movies = len(movie_title)
    random_indices = random.sample(range(total_movies), num_movies)
    return [get_movie_info(movie_title[i]) for i in random_indices]

# 推薦関数の定義
def recommend_movies(user_movies, query, top_k=5):
    query_vector = np.array(client.embeddings.create(input=[query], model='text-embedding-3-large').data[0].embedding)
    vectors = [np.array(id2emb[movie]) for movie in user_movies]
    vectors.append(query_vector)
    average_vector = np.mean(vectors, axis=0)
    distances, indices = index.search(np.array([average_vector]).astype('float32'), top_k)
    recommended_movies = indices[0]
    recommended_movies = recommended_movies[~np.isin(recommended_movies, user_movies)]
    return recommended_movies

# 推薦理由生成のための関数
def generate_recommendation_reason(movie, query):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは優秀な映画ライターです。"},
            {"role": "user", "content": f"以下の映画が推薦されました：{movie_title[movie]}\nユーザが希望していた特徴は「{query}」です。\n{movie_description[movie]}に基づいて、この映画のおすすめポイントをわかりやすく簡潔に2文以内で説明してください。"}
        ],
        temperature=1.2,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

# UIの作成
st.header('Netflixの映画とドラマをおすすめ！', anchor=False)

# セッションステートの初期化
if 'user_movies' not in st.session_state:
    st.session_state['user_movies'] = []
if 'random_movies' not in st.session_state:
    st.session_state['random_movies'] = get_random_movies()

# ランダムボタンの配置
if st.button('ランダム'):
    st.session_state['random_movies'] = get_random_movies()

# ランダムに選択された映画の表示
st.subheader('おすすめの映画・ドラマ')
cols = st.columns(4)
for idx, movie in enumerate(st.session_state['random_movies']):
    if movie:
        with cols[idx % 4]:
            if movie['poster_url'] and movie['poster_url'] != 'N/A':
                st.image(movie['poster_url'], use_column_width='auto')
            st.markdown(f"**[{movie['title']}](https://www.imdb.com/title/{movie['imdb_id']}/)**")
            if st.button(f"選択: {movie['title']}", key=movie['imdb_id']):
                st.session_state['user_movies'].append(movie['title'])
                st.success(f"{movie['title']} を選択しました。")

# ユーザーが選択した映画の表示
if st.session_state['user_movies']:
    st.subheader('あなたが選択した映画・ドラマ')
    st.write(', '.join(st.session_state['user_movies']))

# ユーザーが希望する特徴を入力
query = st.text_input('今みたい映画・ドラマの特徴やジャンルを入力してください')

# おすすめの映画を表示
if st.button('おすすめの映画を表示'):
    if st.session_state['user_movies'] and query:
        user_movie_ids = [np.where(movie_title == movie)[0][0] for movie in st.session_state['user_movies']]
        recommended_movies = recommend_movies(user_movie_ids, query, 10)
        for movie in recommended_movies[:5]:
            title = movie_title[movie]
            movie_info = get_movie_info(title)
            if movie_info:
                st.subheader(movie_info['title'], anchor=False)
                if movie_info['poster_url'] and movie_info['poster_url'] != 'N/A':
                    st.image(movie_info['poster_url'], use_column_width='auto')
                reason = generate_recommendation_reason(movie, query)
                st.markdown(f"**推薦理由:** {reason} [映画詳細ページ](https://www.imdb.com/title/{movie_info['imdb_id']}/)", unsafe_allow_html=True)
    else:
        st.warning('映画を選択し、希望する特徴を入力してください。')
