# test_bert_loading.py - BERT 로딩 테스트
import os
import json

print("BERT 로딩 테스트")

# bert-for-tf2 import 테스트
try:
    import bert
    from bert import BertModelLayer
    from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
    print("✓ bert-for-tf2 라이브러리 로드 성공")
    BERT_AVAILABLE = True
except ImportError as e:
    print(f"✗ bert-for-tf2 라이브러리 로드 실패: {e}")
    BERT_AVAILABLE = False

if BERT_AVAILABLE:
    # 필요한 파일들 확인
    files_to_check = [
        "./pretrained/bert_config.json",
        "./pretrained/model.ckpt-381250.index",
        "./pretrained/model.ckpt-381250.data-00000-of-00001",
        "./pretrained/korpat_vocab.txt"
    ]
    
    print("\n파일 존재 확인:")
    for file_path in files_to_check:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
    
    # 간단한 BERT 모델 로딩 테스트
    try:
        max_len = 256
        
        if os.path.exists("./pretrained/bert_config.json"):
            print("\nbert_config.json에서 설정 로드 시도...")
            with open("./pretrained/bert_config.json", 'r') as f:
                bert_config = json.load(f)
            print(f"설정 키: {list(bert_config.keys())}")
            
            # 기본적인 파라미터만 시도
            bert_params = bert.params_from_pretrained_ckpt("./pretrained/")
            print("✓ BERT 파라미터 로드 성공")
            
            # 모델 레이어 생성 테스트
            bert_layer = BertModelLayer.from_params(bert_params, name="bert")
            print("✓ BERT 레이어 생성 성공")
            
        else:
            print("\nbert_config.json이 없어 기본 설정으로 시도...")
            bert_params = bert.params_from_pretrained_ckpt("./pretrained/")
            print("✓ 기본 BERT 파라미터 로드 성공")
            
    except Exception as e:
        print(f"✗ BERT 모델 로딩 실패: {e}")
        print("상세 오류:", str(e))

else:
    print("BERT 라이브러리가 없어 테스트를 건너뜁니다.")

print("\n테스트 완료")
