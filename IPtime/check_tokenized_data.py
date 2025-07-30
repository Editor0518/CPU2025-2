# check_tokenized_data.py - 토크나이징된 데이터 확인
import pandas as pd
import os

def check_tokenized_data():
    """토크나이징된 특허 데이터 확인"""
    
    pkl_file = "patents_tokenized_with_embeddings.pkl"
    
    if not os.path.exists(pkl_file):
        print(f"파일이 없습니다: {pkl_file}")
        return
    
    try:
        df = pd.read_pickle(pkl_file)
        print(f"[OK] {pkl_file} 로드 완료")
        print(f"데이터 건수: {len(df)}")
        print(f"컬럼: {list(df.columns)}")
        
        # 토크나이징 결과 확인
        if 'input_ids' in df.columns and 'segment_ids' in df.columns:
            print(f"\n=== 토크나이징 결과 확인 ===")
            
            # 샘플 데이터 확인
            sample = df.iloc[0]
            print(f"첫 번째 특허:")
            print(f"  발명명: {str(sample.get('inventionTitle', 'N/A'))[:50]}...")
            print(f"  초록 길이: {len(str(sample.get('astrtCont', '')))}")
            
            if 'tokens' in df.columns:
                tokens = sample['tokens']
                print(f"  토큰 수: {len(tokens)}")
                print(f"  토큰 예시: {tokens[:10]}")
            
            input_ids = sample['input_ids']
            segment_ids = sample['segment_ids']
            print(f"  input_ids 길이: {len(input_ids)}")
            print(f"  input_ids 예시: {input_ids[:10]}")
            print(f"  segment_ids 예시: {segment_ids[:10]}")
            
            # 통계
            print(f"\n=== 전체 통계 ===")
            if 'tokens' in df.columns:
                token_lengths = df['tokens'].apply(len)
                print(f"평균 토큰 수: {token_lengths.mean():.1f}")
                print(f"최대 토큰 수: {token_lengths.max()}")
                print(f"최소 토큰 수: {token_lengths.min()}")
            
            # 빈 데이터 확인
            empty_abstracts = df['astrtCont'].fillna('').apply(lambda x: len(str(x).strip()) == 0).sum()
            print(f"빈 초록 수: {empty_abstracts}")
            
            print(f"\n=== 성공! ===")
            print(f"KorPatBERT 토크나이저를 사용한 특허 데이터 처리가 완료되었습니다.")
            print(f"- 총 {len(df)}건의 특허 데이터")
            print(f"- input_ids, segment_ids로 토크나이징 완료")
            print(f"- 파일: {pkl_file}")
            
        else:
            print("토크나이징 컬럼이 없습니다!")
            
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    check_tokenized_data()
