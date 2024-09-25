import numpy as np
import faiss
import streamlit as st
import zipfile
from openai import OpenAI
import re
import requests

client = OpenAI()
client.api_key = st.secrets["OPENAI_API_KEY"]
omdb_api_ky = st.secrets["OMDB_API_KEY"]

# 映画データの読み込み
zip_file_path = 'movie_info.zip'

with zipfile.ZipFile(zip_file_path, 'r') as z:
    # ZIPファイル内のファイルリストを取得
    file_list = z.namelist()
    file_list = sorted(file_list)

    # CSVファイルを指定して読み込む
    with z.open(file_list[0]) as id2emb:
        id2emb = np.load(id2emb, allow_pickle=True)
    with z.open(file_list[1]) as movie_description:
        movie_description = np.load(movie_description, allow_pickle=True)
    with z.open(file_list[2]) as movie_title:
        movie_title = np.load(movie_title, allow_pickle=True)
    
index = faiss.read_index('movie_index.faiss')


# 推薦関数の定義
def recommend_movies(user_movies, query, top_k=5):
    # クエリをベクトル化
    query_vector = np.array(client.embeddings.create(input = [query], model='text-embedding-3-large').data[0].embedding)

    # # user_moviesの映画のベクトルを加算
    # for movie in user_movies:
    #     query_vector += np.array(id2emb[movie])

    # ベクトルの平均化
    vectors = [np.array(id2emb[movie]) for movie in user_movies]
    vectors.append(query_vector)
    # ベクトルの平均を計算
    average_vector = np.mean(vectors, axis=0)

    # 類似度の高い映画を検索
    distances, indices = index.search(np.array([average_vector]).astype('float32'), top_k)

    recommended_movies = indices[0]

    # ユーザーが選択した映画を除外
    recommended_movies = recommended_movies[~np.isin(recommended_movies, user_movies)]
    
    return recommended_movies

def rerank(user_movies, query):
    movie_desc = [{movie_title[m]: f'index:{m}:概要{movie_description[m]}'} for m in user_movies] 
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは優秀な映画ライターです。"},
            {"role": "user", "content": f"以下の映画が推薦されました：{movie_desc}\nユーザが希望していた$特徴$は「{query}」です。\n$特徴$に合わせて、おすすめ順に映画を並び替えて、**映画のindexのみ**を出力してください。\n出力形式はpythonの配列でお願いします。"}
        ],
        temperature=1.2,
        max_tokens=500,
    )
    list = re.findall(r'\d+', response.choices[0].message.content.strip())
    rerank_index = [int(l) for l in list]
    return rerank_index

# 推薦理由生成のための関数
def generate_recommendation_reason(movie, query):
    # prompt = f"以下の映画が推薦されました：{movie_title[movie]}。" # 埋め込みなしのdfを読み込む手はある
    # # prompt += f"ユーザーが好きな映画は{', '.join(movies_df[movies_df['id'].isin(user_movies)]['title'].tolist())}で、"
    # prompt += f"希望する特徴は「{query}」です。あなたの知識に基づいて、この映画が推薦された理由を2文で説明してください。" # ここにRAGを入れる、ほんとはそのまえに候補を多くしておいてリランクも入れたい
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

# Streamlitを使用したシンプルなUIの作成
st.title('Netflixの映画とドラマをおすすめ！')

# ユーザーが映画を選択
user_movies = st.multiselect('好きな映画・ドラマを選択してください', movie_title)

# ユーザーが希望する特徴を入力
query = st.text_input('今みたい映画・ドラマの特徴やジャンルを入力してください')

if st.button('おすすめの映画を表示'):
    st.write('**あなたが好きな映画・ドラマと今みたい気分にあわせたレコメンド**', )
    if user_movies and query:
        # ユーザーが選択した映画のIDを取得
        user_movie_ids = [np.where(movie_title == movie)[0][0] for movie in user_movies]
        recommended_movies = recommend_movies(user_movie_ids, query, 10)
        recommended_movies = rerank(recommended_movies, query)
        
        for i, movie in enumerate(recommended_movies):
            if i > 4:
                break

            # 映画のタイトルを取得
            title = movie_title[movie]
    
            st.subheader(title, anchor=False)
            
            # OMDB APIを使ってポスター画像を取得
            api_key = '879fb267'  # OMDB APIキーを設定
            params = {
                'apikey': omdb_api_ky,
                't': title
            }
            response = requests.get('http://www.omdbapi.com/', params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('Response') == 'True':
                    poster_url = data.get('Poster')
                    imdb_id = data.get('imdbID')
                    imdb_url = f"https://www.imdb.com/title/{imdb_id}/"
                    if poster_url and poster_url != 'N/A':
                        st.image(poster_url, use_column_width='auto')
                    else:
                        st.write('ポスター画像が見つかりませんでした。')
                else:
                    st.write('映画情報が見つかりませんでした。')
            else:
                st.write('OMDB APIへのリクエストに失敗しました。')
            
            # 推薦理由を表示
            reason = generate_recommendation_reason(movie, query)
            st.markdown(f"**推薦理由:** {reason}[映画詳細ページ]({imdb_url})", unsafe_allow_html=True)
    else:
        st.write('映画を選択し、希望する特徴を入力してください。')