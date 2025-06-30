try:
    import MeCab
    print("MeCab 모듈 import 성공")
    
    # 기본 Tagger 생성 시도
    try:
        tagger = MeCab.Tagger()
        print("기본 Tagger 생성 성공")
        result = tagger.parse("안녕하세요")
        print("파싱 결과:", result)
    except Exception as e:
        print(f"기본 Tagger 실패: {e}")
        
        # 사전 경로 지정하여 시도
        try:
            tagger = MeCab.Tagger('-d ./mecab-ko-dic')
            print("사전 경로 지정 Tagger 생성 성공")
            result = tagger.parse("안녕하세요")
            print("파싱 결과:", result)
        except Exception as e2:
            print(f"사전 경로 지정 Tagger도 실패: {e2}")
            
except ImportError as e:
    print(f"MeCab 모듈 import 실패: {e}")
except Exception as e:
    print(f"기타 오류: {e}")