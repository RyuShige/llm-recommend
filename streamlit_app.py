import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from transformers import pipeline
import streamlit as st

# 仮想的な映画データの作成
movies_data = [
    {"movieId": 1, "title": "宇宙の彼方へ", "genres": "SF、アドベンチャー", "description": "宇宙探査隊の壮大な冒険を描いたSF映画。未知の惑星での予想外の出来事に遭遇する。"},
    {"movieId": 2, "title": "恋は突然に", "genres": "ロマンス、コメディ", "description": "予期せぬ出会いから始まる、心温まるラブストーリー。笑いあり涙ありの感動作。"},
    {"movieId": 3, "title": "闇の刑事", "genres": "犯罪、スリラー", "description": "ベテラン刑事が連続殺人事件に挑む。緊迫感あふれる展開が観客を釘付けにする。"},
    {"movieId": 4, "title": "青春の輝き", "genres": "ドラマ、青春", "description": "高校生たちの夢と挫折、友情を描いた青春ドラマ。若者の成長物語に心打たれる。"},
    {"movieId": 5, "title": "幽霊屋敷の謎", "genres": "ホラー、ミステリー", "description": "古い洋館に住み着いた家族が遭遇する怪奇現象。徐々に明らかになる屋敷の秘密に戦慄する。"},
    {"movieId": 6, "title": "グローバル・クライシス", "genres": "アクション、政治", "description": "世界規模のテロ攻撃に立ち向かう特殊部隊の活躍。国際政治の駆け引きも絡む大作。"},
    {"movieId": 7, "title": "タイムパラドックス", "genres": "SF、ミステリー", "description": "時間旅行によって引き起こされる予想外の結果。過去と未来が交錯する頭脳派SF。"},
    {"movieId": 8, "title": "ラストサムライ", "genres": "歴史、ドラマ", "description": "幕末の日本を舞台に、侍の精神に魅了される外国人武士の物語。文化の衝突と融合を描く。"},
    {"movieId": 9, "title": "ウォール街の狼", "genres": "ドラマ、犯罪", "description": "1990年代のウォール街を舞台に、若き株式ブローカーの栄光と没落を描く実話ベースの作品。"},
    {"movieId": 10, "title": "宇宙戦争2150", "genres": "SF、アクション", "description": "西暦2150年、地球に襲来した異星人との壮絶な戦いを描くSF大作。最新CGIによる圧巻の映像美。"},
    {"movieId": 11, "title": "永遠の約束", "genres": "ロマンス、ドラマ", "description": "50年間離ればなれだった幼なじみの再会と、よみがえる初恋の想い出を描いた感動のラブストーリー。"},
    {"movieId": 12, "title": "ザ・プロフェッショナル", "genres": "アクション、犯罪", "description": "凄腕の殺し屋と少女の異色コンビが、汚職警官に立ち向かう。ハードボイルドなアクション映画。"},
    {"movieId": 13, "title": "ミッドナイト・イン・東京", "genres": "コメディ、ロマンス", "description": "東京を訪れたアメリカ人俳優が、一晩で経験する奇妙で楽しい冒険。文化の違いによる勘違いが笑いを誘う。"},
    {"movieId": 14, "title": "バーチャル・リアリティ", "genres": "SF、スリラー", "description": "仮想現実と現実世界の境界線が曖昧になる近未来。主人公の現実認識が揺らぐサスペンスSF。"},
    {"movieId": 15, "title": "オーシャンズ・レガシー", "genres": "犯罪、コメディ", "description": "腕利きの泥棒たちが、不可能と言われた美術館からの絵画盗難に挑む。軽快な音楽とともに展開するクライムコメディ。"},
    {"movieId": 16, "title": "サバイバル・アイランド", "genres": "アドベンチャー、ドラマ", "description": "無人島に漂着した群衆が、協力して生き抜く姿を描くサバイバルドラマ。人間性の光と影が浮き彫りに。"},
    {"movieId": 17, "title": "バック・トゥ・ザ・80's", "genres": "コメディ、SF", "description": "事故により1980年代にタイムスリップした現代の高校生。カルチャーショックと珍事件の連続に爆笑必至。"},
    {"movieId": 18, "title": "エターナル・ソング", "genres": "ミュージカル、ドラマ", "description": "音楽の才能に恵まれた少女の、夢を追う感動の物語。心に響く歌声と感動のストーリーが融合。"},
    {"movieId": 19, "title": "ダーク・ウェブ", "genres": "スリラー、犯罪", "description": "インターネットの闇、ダークウェブを舞台にしたサイバー犯罪スリラー。テクノロジーの光と影を鋭く描写。"},
    {"movieId": 20, "title": "グレート・エスケープ", "genres": "アクション、歴史", "description": "第二次世界大戦中のドイツ捕虜収容所からの大脱走計画を描いた実話ベースの作品。緊迫感あふれる脱出劇に息をのむ。"},
    {"movieId": 21, "title": "銀河鉄道の夜", "genres": "アニメ、ファンタジー", "description": "宮沢賢治の名作を元にしたファンタジーアニメ。美しい映像と哲学的なテーマが大人も魅了する。"},
    {"movieId": 22, "title": "アンダーカバー", "genres": "犯罪、ドラマ", "description": "犯罪組織に潜入した警官の過酷な日々を描く。正義と罪の狭間で揺れ動く人間ドラマ。"},
    {"movieId": 23, "title": "ラスト・サマー", "genres": "青春、ドラマ", "description": "高校卒業後、それぞれの道を歩み始める前の最後の夏休みを描いた青春ドラマ。若者の希望と不安が交錯する。"},
    {"movieId": 24, "title": "インフィニティ・ウォーズ", "genres": "SF、アクション", "description": "銀河の覇権をめぐる壮大な宇宙戦争を描くSF大作。複雑な人種関係と政治的駆け引きも見どころ。"},
    {"movieId": 25, "title": "パラレルワールド", "genres": "SF、ロマンス", "description": "平行世界を行き来できるようになった主人公。別の世界線の自分や恋人との出会いが、人生の選択を問い直させる。"},
    {"movieId": 26, "title": "ゴースト・ハウス", "genres": "ホラー、ミステリー", "description": "霊能力者一家が家庭内の問題を解決しながら、依頼された心霊現象を解決していく。ホラーとヒューマンドラマが融合。"},
    {"movieId": 27, "title": "グランド・ホテル", "genres": "ドラマ、ミステリー", "description": "高級ホテルを舞台に、宿泊客と従業員の人間模様を描くオムニバスドラマ。様々な人生の交差点を鮮やかに描写。"},
    {"movieId": 28, "title": "エコ・ウォリアーズ", "genres": "ドキュメンタリー、アドベンチャー", "description": "環境保護のために戦う活動家たちの姿を追ったドキュメンタリー。美しい自然映像と人間ドラマが胸を打つ。"},
    {"movieId": 29, "title": "コード・レッド", "genres": "アクション、スリラー", "description": "テロリストに占拠された高層ビルからの脱出劇。主人公の機転と勇気が、多くの命を救う。"},
    {"movieId": 30, "title": "ミラクル・ガーデン", "genres": "ファンタジー、ドラマ", "description": "都会の片隅にある不思議な庭を通じて、人々の傷ついた心が癒されていく物語。ほのぼのとした雰囲気が心を和ませる。"}
]

movies_df = pd.DataFrame(movies_data)

# 映画の特徴量を作成（ジャンルとタイトルと説明を結合）
movies_df['features'] = movies_df['genres'] + ' ' + movies_df['title'] + ' ' + movies_df['description']

# Sentence Transformerモデルの読み込み
model = SentenceTransformer('all-MiniLM-L6-v2')

# 映画の特徴量をベクトル化
movie_embeddings = model.encode(movies_df['features'].tolist())

# FAISSインデックスの作成
dimension = movie_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(movie_embeddings.astype('float32'))

# 推薦関数の定義
def recommend_movies(user_movies, desired_features, top_k=5):
    # ユーザーが選択した映画と希望する特徴を結合
    query = ' '.join(movies_df[movies_df['movieId'].isin(user_movies)]['features'].tolist() + [desired_features])
    
    # クエリをベクトル化
    query_vector = model.encode([query])
    
    # 類似度の高い映画を検索
    distances, indices = index.search(query_vector.astype('float32'), top_k)
    
    recommended_movies = movies_df.iloc[indices[0]]
    
    return recommended_movies

# 推薦理由生成のための関数
def generate_recommendation_reason(movie, user_movies, desired_features):
    prompt = f"以下の映画が推薦されました：{movie['title']}。この映画のジャンルは{movie['genres']}です。"
    prompt += f"ユーザーが好きな映画は{', '.join(movies_df[movies_df['movieId'].isin(user_movies)]['title'].tolist())}で、"
    prompt += f"希望する特徴は「{desired_features}」です。この映画が推薦された理由を2文で説明してください。"

    generator = pipeline('text-generation', model='gpt2')
    reason = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    return reason.strip()

# Streamlitを使用したシンプルなUIの作成
st.title('映画推薦システム')

# ユーザーが映画を選択
user_movies = st.multiselect('好きな映画を選択してください', movies_df['title'].tolist())

# ユーザーが希望する特徴を入力
desired_features = st.text_input('見たい映画の特徴やジャンルを入力してください')

if st.button('おすすめの映画を表示'):
    if user_movies and desired_features:
        user_movie_ids = movies_df[movies_df['title'].isin(user_movies)]['movieId'].tolist()
        recommended_movies = recommend_movies(user_movie_ids, desired_features)
        
        for _, movie in recommended_movies.iterrows():
            st.subheader(movie['title'])
            st.write(f"ジャンル: {movie['genres']}")
            st.write(f"説明: {movie['description']}")
            reason = generate_recommendation_reason(movie, user_movie_ids, desired_features)
            st.write(f"推薦理由: {reason}")
    else:
        st.write('映画を選択し、希望する特徴を入力してください。')