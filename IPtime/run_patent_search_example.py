# run_patent_search_example.py - 예시 실행 스크립트
import subprocess
import sys
import os

def run_patent_search():
    """patent_search.py를 예시 데이터로 실행"""
    
    print("특허 검색 예시 실행")
    print("=" * 50)
    
    # 테스트용 입력값 준비
    search_word = "(생성형+생성적+generative+generating)*머신러닝*(인공지능+AI+ARTIFICIAL INTELLIGENCE)*신경망*신호*(언어^1모델)*IPC=[G06N]*PD=[20100101~20250628]"  # 검색어
    withdrawn_flag = "n"    # 취하/소멸/포기 포함 안함
    
    print(f"검색어: {search_word}")
    print(f"취하/소멸/포기 포함: {withdrawn_flag}")
    print()
    
    # patent_search.py 실행을 위한 입력 준비
    input_data = f"{search_word}\n{withdrawn_flag}\n"
    
    try:
        # subprocess를 사용하여 patent_search.py 실행
        result = subprocess.run(
            [sys.executable, "patent_search.py"],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=60  # 60초 타임아웃
        )
        
        print("=== 실행 결과 ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== 오류 메시지 ===")
            print(result.stderr)
            
        print(f"실행 완료 (반환 코드: {result.returncode})")
        
        # 생성된 파일 확인
        if os.path.exists("patents_tokenized.pkl"):
            import pandas as pd
            df = pd.read_pickle("patents_tokenized.pkl")
            print(f"\n생성된 파일 정보:")
            print(f"- 파일명: patents_tokenized.pkl")
            print(f"- 데이터 건수: {len(df)}")
            print(f"- 컬럼: {list(df.columns)}")
            
    except subprocess.TimeoutExpired:
        print("실행 시간 초과 (60초)")
    except Exception as e:
        print(f"실행 오류: {e}")

if __name__ == "__main__":
    run_patent_search()
