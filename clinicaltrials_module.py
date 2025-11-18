import requests
import pandas as pd
from time import sleep

# =========================================
# 1) clinicaltrials.gov에서 단일 약물 검색
# =========================================
def search_clinical_trials(drug_name, cancer_type=None, max_results=200):
    """
    clinicaltrials.gov API v2를 사용해 특정 약물에 대한 임상시험 정보 조회
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    query = drug_name
    if cancer_type:
        query = f"{drug_name} AND {cancer_type}"

    params = {
        "query.term": query,
        "pageSize": min(max_results, 1000),
        "format": "json"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Research Project)"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f"⚠️ HTTP {response.status_code} for {drug_name}")
            return pd.DataFrame(columns=["drug", "nct_id", "title", "condition", "phase", "status"])
        
        data = response.json()
        
    except requests.exceptions.Timeout:
        print(f"⚠️ Timeout for {drug_name}")
        return pd.DataFrame(columns=["drug", "nct_id", "title", "condition", "phase", "status"])
    except requests.exceptions.JSONDecodeError as e:
        print(f"⚠️ JSON decode error for {drug_name}")
        return pd.DataFrame(columns=["drug", "nct_id", "title", "condition", "phase", "status"])
    except Exception as e:
        print(f"⚠️ Error fetching for {drug_name}: {e}")
        return pd.DataFrame(columns=["drug", "nct_id", "title", "condition", "phase", "status"])

    # API v2 응답 구조 파싱
    studies = data.get("studies", [])
    
    if not studies:
        # 검색 결과가 없는 경우 (에러 아님)
        return pd.DataFrame(columns=["drug", "nct_id", "title", "condition", "phase", "status"])

    rows = []
    for s in studies:
        protocol = s.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        design_module = protocol.get("designModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        
        # Phase 정보 추출 (리스트를 쉼표로 구분)
        phases = design_module.get("phases", [])
        phase_str = ", ".join(phases) if phases else ""
        
        # Condition 정보 추출
        conditions = conditions_module.get("conditions", [])
        condition_str = ", ".join(conditions) if conditions else ""
        
        rows.append({
            "drug": drug_name.upper(),
            "nct_id": id_module.get("nctId", ""),
            "title": id_module.get("briefTitle", ""),
            "condition": condition_str,
            "phase": phase_str,
            "status": status_module.get("overallStatus", ""),
        })

    df = pd.DataFrame(rows)
    print(f"✅ {drug_name}: {len(df)} trials found")
    return df

# =========================================
# 2) 임상시험 대표 Trial 선택 (최고 Phase)
# =========================================
def extract_top_trial(trial_df):
    """
    하나의 약물에 대한 임상시험 DataFrame에서
    '가장 높은 Phase'의 trial 한 개를 선택.
    """

    if trial_df.empty:
        return {
            "top_nct_id": None,
            "top_phase": None,
            "top_title": None
        }

    # Phase 우선순위 정의
    phase_priority = {
        "PHASE 4": 4,
        "PHASE 3": 3,
        "PHASE 2": 2,
        "PHASE 1": 1
    }

    # Phase score 계산
    def score_phase(p):
        if not p:
            return 0
        p = p.upper().strip()
        return phase_priority.get(p, 0)

    trial_df = trial_df.copy()
    trial_df["phase_score"] = trial_df["phase"].apply(score_phase)

    # 최고 Phase 정렬
    top_row = trial_df.sort_values("phase_score", ascending=False).iloc[0]

    return {
        "top_nct_id": top_row["nct_id"],
        "top_phase": top_row["phase"],
        "top_title": top_row["title"]
    }



# =========================================
# 3) 여러 약물에 대해 임상시험 검색 후 TSV 저장
# =========================================
def run_clinical_trials(drug_list, cancer_type, output_path):
    """
    약물 리스트 전체에 대해 clinicaltrials.gov 검색 수행 후 TSV 저장
    
    Parameters:
        drug_list: 예) ["ERLOTINIB", "GEFITINIB"]
        cancer_type: 필터링용 암종명 (string)
        output_path: TSV 저장 경로
    
    Returns:
        DataFrame(모든 trial rows)
    """

    print(f"🔍 clinicaltrials.gov 검색 시작 (약물 {len(drug_list)}개)")

    all_rows = []

    for drug in drug_list:
        print(f"   → {drug} 검색 중...")

        df = search_clinical_trials(
            drug_name=drug,
            cancer_type=cancer_type,
            max_results=200
        )

        if not df.empty:
            all_rows.append(df)

        sleep(1)  # API overload 방지
    
    if all_rows:
        result_df = pd.concat(all_rows, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=["drug", "nct_id", "title", "condition", "phase", "status"])

    # TSV 저장
    result_df.to_csv(output_path, sep="\t", index=False)
    print(f"📄 ClinicalTrials TSV 저장 완료: {output_path}")

    return result_df



# =========================================
# 4) 약물별 대표 임상시험 요약 테이블 생성
# =========================================
def build_trial_summary_table(clinical_df):
    """
    전체 임상시험 DF에서 약물별 대표 임상시험 정보 테이블 생성

    Returns:
        DataFrame(columns=[
            drug, n_clinical_trials, top_nct_id, top_phase, top_title
        ])
    """

    summary_rows = []

    for drug, df in clinical_df.groupby("drug"):
        top = extract_top_trial(df)
        summary_rows.append({
            "drug": drug,
            "n_clinical_trials": df["nct_id"].nunique(),
            "top_nct_id": top["top_nct_id"],
            "top_phase": top["top_phase"],
            "top_title": top["top_title"]
        })

    return pd.DataFrame(summary_rows)



# =========================================
# 2) 임상시험 대표 Trial 선택 (최고 Phase)
# =========================================
def extract_top_trial(trial_df):
    """
    하나의 약물에 대한 임상시험 DataFrame에서
    '가장 높은 Phase'의 trial 한 개를 선택.
    """

    if trial_df.empty:
        return {
            "top_nct_id": None,
            "top_phase": None,
            "top_title": None
        }

    # Phase 우선순위 정의
    phase_priority = {
        "PHASE 4": 4,
        "PHASE 3": 3,
        "PHASE 2": 2,
        "PHASE 1": 1
    }

    # Phase score 계산
    def score_phase(p):
        if not p:
            return 0
        p = p.upper().strip()
        return phase_priority.get(p, 0)

    trial_df = trial_df.copy()
    trial_df["phase_score"] = trial_df["phase"].apply(score_phase)

    # 최고 Phase 정렬
    top_row = trial_df.sort_values("phase_score", ascending=False).iloc[0]

    return {
        "top_nct_id": top_row["nct_id"],
        "top_phase": top_row["phase"],
        "top_title": top_row["title"]
    }



# =========================================
# 3) 여러 약물에 대해 임상시험 검색 후 TSV 저장
# =========================================
def run_clinical_trials(drug_list, cancer_type, output_path):
    """
    약물 리스트 전체에 대해 clinicaltrials.gov 검색 수행 후 TSV 저장
    
    Parameters:
        drug_list: 예) ["ERLOTINIB", "GEFITINIB"]
        cancer_type: 필터링용 암종명 (string)
        output_path: TSV 저장 경로
    
    Returns:
        DataFrame(모든 trial rows)
    """

    print(f"🔍 clinicaltrials.gov 검색 시작 (약물 {len(drug_list)}개)")

    all_rows = []

    for drug in drug_list:
        print(f"   → {drug} 검색 중...")

        df = search_clinical_trials(
            drug_name=drug,
            cancer_type=cancer_type,
            max_results=200
        )

        if not df.empty:
            all_rows.append(df)

        sleep(0.2)  
    
    if all_rows:
        result_df = pd.concat(all_rows, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=["drug", "nct_id", "title", "condition", "phase", "status"])

    # TSV 저장
    result_df.to_csv(output_path, sep="\t", index=False)
    print(f"📄 ClinicalTrials TSV 저장 완료: {output_path}")

    return result_df



# =========================================
# 4) 약물별 대표 임상시험 요약 테이블 생성
# =========================================
def build_trial_summary_table(clinical_df):
    """
    전체 임상시험 DF에서 약물별 대표 임상시험 정보 테이블 생성

    Returns:
        DataFrame(columns=[
            drug, n_clinical_trials, top_nct_id, top_phase, top_title
        ])
    """

    summary_rows = []

    for drug, df in clinical_df.groupby("drug"):
        top = extract_top_trial(df)
        summary_rows.append({
            "drug": drug,
            "n_clinical_trials": df["nct_id"].nunique(),
            "top_nct_id": top["top_nct_id"],
            "top_phase": top["top_phase"],
            "top_title": top["top_title"]
        })

    return pd.DataFrame(summary_rows)