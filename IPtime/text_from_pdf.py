import os
import re
import pdfplumber  # pip install pdfplumber

def extract_text_from_pdf(pdf_path):
    all_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + '\n'
    return all_text

def extract_metadata(text):
    metadata = {}

    # 문서 상태 및 번호 구분
    if "등록특허공보" in text or re.search(r'\(11\)\s*등록번호', text):
        metadata['상태'] = "등록"
        metadata['등록번호'] = re.search(r'\(11\)\s*등록번호\s+([\d\-]+)', text)
        metadata['공고일자'] = re.search(r'\(45\)\s*공고일자\s*([0-9년월일\s]+)', text)
        metadata['등록일자'] = re.search(r'\(24\)\s*등록일자\s*([0-9년월일\s]+)', text)
    else:
        metadata['상태'] = "공개"
        metadata['공개번호'] = re.search(r'\(11\)\s*공개번호\s+([\d\-]+)', text)
        metadata['공개일자'] = re.search(r'\(43\)\s*공개일자\s*([0-9년월일\s]+)', text)

    metadata['출원번호'] = re.search(r'\(21\)\s*출원번호\s+([\d\-]+)', text)
    metadata['출원일자'] = re.search(r'\(22\)\s*출원일자\s+([0-9년월일\s]+)', text)

    # IPC 및 CPC 분류
    ipc_match = re.search(r'\(51\)[^\n]*\n((?:.*\n)*?)\(52\)', text)
    metadata['IPC'] = re.findall(r'[A-Z]\d{2}[A-Z]?\s*\d+/\d+', ipc_match.group(1)) if ipc_match else []

    cpc_match = re.search(r'\(52\)\s*CPC특허분류\s*((?:.*\n)*?)\(21\)', text)
    metadata['CPC'] = re.findall(r'[A-Z]\d{2}[A-Z]?\s*\d+/\d+', cpc_match.group(1)) if cpc_match else []

    # 발명의 명칭
    title_match = re.search(r'\(54\)\s*발명의 명칭\s*(.+)', text)
    metadata['발명의 명칭'] = title_match.group(1).strip() if title_match else None

    # 요약
    summary_match = re.search(r'\(57\)\s*요 약\s*(.+?)(\n\n|전체 청구항 수|도 \d)', text, re.DOTALL)
    metadata['요약'] = summary_match.group(1).strip() if summary_match else None

    # 청구항들
    claims = re.findall(r'청구항\s*\d+\s*(.*?)\n(?=청구항\s*\d+|\Z)', text, re.DOTALL)
    metadata['청구범위'] = [c.strip() for c in claims]

    # 청구항 수
    total_claims = re.search(r'전체 청구항 수\s*:\s*총\s*(\d+)\s*항', text)
    metadata['청구항 수'] = int(total_claims.group(1)) if total_claims else len(claims)

    # 기술분야
    tech_match = re.search(r'기\s*술\s*분\s*야\s*(.+?)(배\s*경\s*기\s*술|도면의 간단한 설명)', text, re.DOTALL)
    metadata['기술분야'] = tech_match.group(1).strip() if tech_match else None

    return {k: (v.group(1).strip() if isinstance(v, re.Match) else v) for k, v in metadata.items()}

def process_all_pdfs_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            print(f"\n📂 파일명: {filename}")
            print("=" * 100)
            text = extract_text_from_pdf(pdf_path)
            info = extract_metadata(text)
            for key, val in info.items():
                print(f"📌 {key}:\n{val}\n{'-'*80}")

# 사용 예시
pdf_directory = './patent-pdf'
process_all_pdfs_in_directory(pdf_directory)
