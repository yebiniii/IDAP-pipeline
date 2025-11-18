import os
import subprocess
import pandas as pd
import re
import sys

# -------------------------------
# 0) OFFICIAL ONCOKB CANCER TYPE MAP
# -------------------------------
ONCOKB_CANCER_MAP = {
    "lung cancer": "Non-Small Cell Lung Cancer",
    "nsclc": "Non-Small Cell Lung Cancer",
    "luad": "Lung Adenocarcinoma",
    "lusc": "Lung Squamous Cell Carcinoma",

    "glioblastoma": "Glioblastoma Multiforme",
    "gbm": "Glioblastoma Multiforme",

    "pancreatic cancer": "Pancreatic Adenocarcinoma",
    "pdac": "Pancreatic Adenocarcinoma",

    "breast cancer": "Breast Invasive Carcinoma",
    "brca": "Breast Invasive Carcinoma",

    "colon cancer": "Colon Adenocarcinoma",
    "coad": "Colon Adenocarcinoma",
    "crc": "Colon Adenocarcinoma",
}


def normalize_tumor_type(t: str) -> str:
    key = t.strip().lower()
    return ONCOKB_CANCER_MAP.get(key, t)


# -------------------------------
# 1) RUN ONCOKB ANNOTATOR (개선됨)
# -------------------------------
def run_oncokb(
    maf_path: str,
    cancer_type: str,
    output_maf_path: str,
    oncokb_api_token: str,
    genome_build: str = "GRCh38",
    annotator_path: str = "",
    log_path: str = "annotator.log"
):
    tumor_type = normalize_tumor_type(cancer_type)

    cmd = [
        "python",
        annotator_path,
        "-i", maf_path,
        "-o", output_maf_path,
        "-b", oncokb_api_token,
        "-r", genome_build,
        "-t", tumor_type
    ]

    print("🚀 Running OncoKB Annotator:")
    print(" ".join(cmd))
    print(f"   Tumor Type: {tumor_type}")

    # ✅ 개선된 에러 처리
    try:
        # stderr와 stdout을 모두 캡처
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True,
            text=True,
            timeout=600  # 10분 타임아웃
        )
        
        # 로그 파일에 저장
        with open(log_path, "w") as logf:
            logf.write("=== STDOUT ===\n")
            logf.write(result.stdout)
            logf.write("\n=== STDERR ===\n")
            logf.write(result.stderr)
        
        # 화면에도 출력
        if result.stdout:
            print("📝 OncoKB Output:")
            print(result.stdout[:500])  # 처음 500자만
        
        print(f"✅ OncoKB annotation complete → {output_maf_path}")
        
    except subprocess.CalledProcessError as e:
        print("❌ OncoKB Annotator FAILED!")
        print(f"   Return code: {e.returncode}")
        print(f"   STDOUT: {e.stdout}")
        print(f"   STDERR: {e.stderr}")
        
        # 로그 파일에 에러 저장
        with open(log_path, "w") as logf:
            logf.write(f"ERROR: Return code {e.returncode}\n")
            logf.write("=== STDOUT ===\n")
            logf.write(e.stdout)
            logf.write("\n=== STDERR ===\n")
            logf.write(e.stderr)
        
        raise Exception(f"OncoKB annotator failed. Check {log_path} for details.")
    
    except subprocess.TimeoutExpired:
        print("❌ OncoKB Annotator TIMEOUT (>10 minutes)")
        raise Exception("OncoKB annotator timeout")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

    # ✅ 출력 파일 존재 확인
    if not os.path.exists(output_maf_path):
        raise FileNotFoundError(f"OncoKB output file not created: {output_maf_path}")
    
    # ✅ 파일 크기 확인
    file_size = os.path.getsize(output_maf_path)
    if file_size == 0:
        raise ValueError(f"OncoKB output file is empty: {output_maf_path}")
    
    print(f"   Output file size: {file_size:,} bytes")
    
    return output_maf_path


# -------------------------------
# 2) ROBUST DRUG PARSER
# -------------------------------
def parse_oncokb_drugs(text: str):
    """
    Robust parser for OncoKB drug entries.

    Handles cases like:
      - Osimertinib (Level 1)
      - Trametinib + Dabrafenib
      - Osimertinib; Erlotinib
      - Multi-line entries
    """

    # Split by semicolon OR newline
    items = re.split(r";|\n", str(text))

    drugs = []
    for item in items:
        item = item.strip()

        # Extract drug name before parentheses
        # ex) "Trametinib + Dabrafenib (FDA 1)" → "Trametinib + Dabrafenib"
        match = re.match(r'^(.+?)(?:\s*\(|$)', item)
        if not match:
            continue

        raw_drug = match.group(1).strip()
        if not raw_drug:
            continue

        # 🔥 Combination therapy split
        #   "Trametinib + Dabrafenib" → ["Trametinib", "Dabrafenib"]
        combo = re.split(r'\s*\+\s*', raw_drug)
        for d in combo:
            d = d.strip()
            if d:
                drugs.append(d)

    # unique list
    return list(set(drugs))


# -------------------------------
# 3) EXTRACT DRUGS FROM LEVEL_* COLUMNS (디버깅 강화)
# -------------------------------
def extract_drugs_from_level_columns(maf: pd.DataFrame) -> pd.DataFrame:

    # Find all LEVEL_* columns
    level_cols = [c for c in maf.columns if c.startswith("LEVEL_")]

    if not level_cols:
        print("⚠️ No LEVEL_* columns found. Columns available:")
        print(maf.columns.tolist())
        return pd.DataFrame()

    print(f"   Found LEVEL columns: {level_cols}")

    # ✅ 디버깅: 각 LEVEL 컬럼에 데이터가 있는지 확인
    for col in level_cols:
        non_empty = maf[col].notna().sum()
        if non_empty > 0:
            print(f"   📊 {col}: {non_empty} non-empty entries")
            # 첫 번째 non-empty 값 출력
            first_val = maf[maf[col].notna()][col].iloc[0]
            print(f"      Example: {repr(first_val)[:100]}")

    results = []
    row_count = 0

    for idx, row in maf.iterrows():
        hugo = row.get("Hugo_Symbol", "")
        protein_change = row.get("ONCOKB_PROTEIN_CHANGE", "")

        if pd.isna(protein_change) or protein_change == "":
            variant = hugo
        else:
            variant = f"{hugo} {protein_change}"

        for col in level_cols:
            val = row.get(col)
            if pd.isna(val) or val == "":
                continue

            # ✅ 디버깅: 파싱 전 값 출력
            if row_count < 3:  # 처음 3개만
                print(f"\n   🔍 Row {idx}, {col}:")
                print(f"      Raw value: {repr(val)[:200]}")
            
            drugs = parse_oncokb_drugs(val)
            
            if row_count < 3 and drugs:
                print(f"      Parsed drugs: {drugs}")
            
            row_count += 1

            for drug in drugs:
                results.append({
                    "variant": variant,
                    "drug": drug.upper(),
                    "oncokb_level": col.replace("LEVEL_", ""),
                    "Hugo_Symbol": hugo,
                    "protein_change": protein_change,
                    "oncogenic": row.get("ONCOGENIC", ""),
                    "mutation_effect": row.get("MUTATION_EFFECT", ""),
                    "highest_level": row.get("HIGHEST_LEVEL", "")
                })

    print(f"\n   ✅ Extracted {len(results)} drug-variant associations from {row_count} entries")
    return pd.DataFrame(results)


# -------------------------------
# 4) LOAD ANNOTATED MAF AND EXTRACT TABLE
# -------------------------------
def load_oncokb_table(annotated_maf_path: str) -> pd.DataFrame:

    if not os.path.exists(annotated_maf_path):
        raise FileNotFoundError(f"Annotated MAF not found: {annotated_maf_path}")

    try:
        maf = pd.read_csv(annotated_maf_path, sep="\t", comment="#", low_memory=False)
        print(f"📄 Annotated MAF loaded: {len(maf)} variants, {len(maf.columns)} columns")
    except Exception as e:
        print(f"❌ Error reading annotated MAF: {e}")
        raise

    # ✅ 컬럼 확인
    print(f"   Sample columns: {maf.columns[:10].tolist()}")

    df = extract_drugs_from_level_columns(maf)

    if df.empty:
        print("⚠️ No drugs extracted from LEVEL_* columns.")
        print("   This may be normal if no actionable variants were found.")

    return df


# -------------------------------
# 5) WRAPPER: RUN + PARSE + SAVE
# -------------------------------
def run_oncokb_and_extract(
    maf_path: str,
    cancer_type: str,
    output_tsv_path: str,
    annotator_path: str,
    oncokb_api_token: str,
    genome_build: str = "GRCh38",
    log_path: str = "annotator.log"
):
    
    print("\n" + "="*60)
    print("OncoKB Annotation Module")
    print("="*60)
    print(f"Input MAF: {maf_path}")
    print(f"Cancer Type (raw): {cancer_type}")
    print(f"Cancer Type (normalized): {normalize_tumor_type(cancer_type)}")
    print(f"Genome Build: {genome_build}")
    print("="*60 + "\n")

    # ✅ 입력 파일 확인
    if not os.path.exists(maf_path):
        raise FileNotFoundError(f"Input MAF not found: {maf_path}")
    
    # ✅ Annotator 스크립트 확인
    if not os.path.exists(annotator_path):
        raise FileNotFoundError(f"OncoKB annotator not found: {annotator_path}")

    output_maf_path = output_tsv_path.replace(".tsv", ".annotated.maf")

    try:
        # Run annotator
        run_oncokb(
            maf_path=maf_path,
            cancer_type=cancer_type,
            output_maf_path=output_maf_path,
            oncokb_api_token=oncokb_api_token,
            genome_build=genome_build,
            annotator_path=annotator_path,
            log_path=log_path
        )

        # Extract drugs
        df = load_oncokb_table(output_maf_path)

        # Save output
        if not df.empty:
            df.to_csv(output_tsv_path, sep="\t", index=False)
            print(f"📄 Output TSV saved → {output_tsv_path}")
        else:
            # 빈 결과도 저장
            pd.DataFrame(columns=[
                "variant", "drug", "oncokb_level", "Hugo_Symbol",
                "protein_change", "oncogenic", "mutation_effect", "highest_level"
            ]).to_csv(output_tsv_path, sep="\t", index=False)
            print(f"⚠️ No drugs found, empty TSV saved → {output_tsv_path}")

        return df
    
    except Exception as e:
        print(f"\n❌ OncoKB module failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print(f"\n💡 Check the log file: {log_path}")
        print(f"💡 Verify:")
        print(f"   1. OncoKB API token is valid")
        print(f"   2. MAF file format is correct")
        print(f"   3. Network connection is available")
        print(f"   4. Annotator script path: {annotator_path}")
        
        # 빈 DataFrame 반환 (파이프라인이 계속 진행되도록)
        return pd.DataFrame()