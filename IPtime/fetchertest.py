# main.py

#이거 쓰지마


from patent_fetcher import fetch_patents
from korpat_tokenizer import Tokenizer  # KorPat Tokenizer 클래스가 정의된 파일명에 맞춰 조정하세요
import pandas as pd

def encode_patents(df: pd.DataFrame,
                   text_field: str = 'astrtCont',
                   vocab_path: str = './pretrained/korpat_vocab.txt',
                   max_len: int = 256) -> pd.DataFrame:
    """
    KorPat Tokenizer를 이용해 df[text_field]를 BERT 입력용 토큰 ID, 세그먼트 ID로 변환하여
    'input_ids', 'segment_ids' 컬럼을 추가한 DataFrame을 반환합니다.
    """
    # Tokenizer 초기화 (한글이므로 cased=True)
    tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)

    # 각 특허 텍스트마다 tokenize → (token_ids, segment_ids)
    encodings = df[text_field].fillna('').apply(
        lambda txt: tokenizer.encode(txt, max_len=max_len)
    )

    # 결과를 두 개의 별도 컬럼으로 분리
    df['input_ids'], df['segment_ids'] = zip(*encodings)
    return df

if __name__ == "__main__":
    # 1) 특허 데이터 불러오기
    df = fetch_patents(
        search_word="(생성형 ai)*(게임 스토리)",
        year_to_search="0",  # 전체 연도
    )

    # 2) KorPat Tokenizer로 인코딩
    df = encode_patents(
        df,
        text_field='astrtCont',               # 본문(초록) 대신 inventionTitle 등 다른 필드를 써도 됩니다
        vocab_path='./pretrained/korpat_vocab.txt',
        max_len=256
    )

    # 3) 결과 확인 및 저장
    print(df[['astrtCont', 'input_ids', 'segment_ids']].head())
    # 원하시면 TSV나 pickle 등으로 저장
    df.to_pickle('./patent-data/patents_with_bert.pkl')
    print("✔ BERT 입력용 토큰이 추가된 DataFrame을 저장했습니다.")
