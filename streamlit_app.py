import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from transformers import pipeline
import streamlit as st
import zipfile
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
client.api_key = st.secrets["OPENAI_API_KEY"]

# 映画データの読み込み
zip_file_path = 'movie_info.zip'

with zipfile.ZipFile(zip_file_path, 'r') as z:
    # ZIPファイル内のファイルリストを取得
    file_list = z.namelist() # ['title.npy', 'id_to_emb.npy']
    file_list = sorted(file_list)

    # CSVファイルを指定して読み込む
    with z.open(file_list[0]) as id2emb:
        id2emb = np.load(id2emb, allow_pickle=True)
    with z.open(file_list[1]) as movie_title:
        movie_title = np.load(movie_title, allow_pickle=True)
    
index = faiss.read_index('movie_index.faiss')


# 推薦関数の定義
def recommend_movies(user_movies, query, top_k=5):
    # クエリをベクトル化
    query_vector = np.array(client.embeddings.create(input = [query], model='text-embedding-3-large').data[0].embedding)

    # user_moviesの映画のベクトルを加算
    for movie in user_movies:
        query_vector += np.array(id2emb[movie])

    print(f"クエリベクトルの次元: {np.array([query_vector]).shape}")
    print(f"インデックスの次元: {index.d}")

    # 類似度の高い映画を検索
    distances, indices = index.search(np.array([query_vector]).astype('float32'), top_k)

    recommended_movies = indices[0]

    print(f"推薦された映画のインデックス: {recommended_movies}")

    # ユーザーが選択した映画を除外
    recommended_movies = recommended_movies[~np.isin(recommended_movies, user_movies)]
    
    print(f"ユーザーが選択した映画を除外した映画のインデックス: {recommended_movies}")

    return recommended_movies

# 推薦理由生成のための関数
def generate_recommendation_reason(movie, user_movies, query):
    # prompt = f"以下の映画が推薦されました：{movie_title[movie]}。" # 埋め込みなしのdfを読み込む手はある
    # # prompt += f"ユーザーが好きな映画は{', '.join(movies_df[movies_df['id'].isin(user_movies)]['title'].tolist())}で、"
    # prompt += f"希望する特徴は「{query}」です。あなたの知識に基づいて、この映画が推薦された理由を2文で説明してください。" # ここにRAGを入れる、ほんとはそのまえに候補を多くしておいてリランクも入れたい

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは優秀な映画ライターです。"},
            {"role": "user", "content": f"以下の映画が推薦されました：{movie_title[movie]}\nユーザが希望していた特徴は「{query}」です。\nあなたの知識に基づいて、この映画が推薦された理由を3文で説明してください。"}
        ],
        temperature=1.2,
        max_tokens=500,
    )
    
    return response.choices[0].message.content.strip()

# Streamlitを使用したシンプルなUIの作成
st.title('映画推薦システム')

# ユーザーが映画を選択
user_movies = st.multiselect('好きな映画を選択してください', movie_title)

# ユーザーが希望する特徴を入力
query = st.text_input('見たい映画の特徴やジャンルを入力してください')

if st.button('おすすめの映画を表示'):
    if user_movies and query:
        # ユーザーが選択した映画のIDを取得
        user_movie_ids = [np.where(movie_title == movie)[0][0] for movie in user_movies]
        recommended_movies = recommend_movies(user_movie_ids, query, 10)
        
        for i, movie in enumerate(recommended_movies):
            st.subheader(movie_title[movie])
            # st.write(f"ジャンル: {movie['genres']}")
            # st.write(f"説明: {movie['description']}")
            reason = generate_recommendation_reason(movie, user_movie_ids, query)
            st.write(f"推薦理由: {reason}")
            if i > 3:
                break
    else:
        st.write('映画を選択し、希望する特徴を入力してください。')

# 埋め込み可視化したいな