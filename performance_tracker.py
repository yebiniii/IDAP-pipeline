import time
import psutil
import os
import pandas as pd
from collections import Counter

class PerformanceTracker:
    """
    파이프라인 실행 중 성능 지표를 추적하는 클래스
    """
    
    def __init__(self, patient_id=None):
        self.patient_id = patient_id or "unknown"
        self.start_time = None
        self.runtimes = {}
        self.api_calls = {
            "oncokb": 0,
            "pubmed": 0,
            "clinicaltrials": 0
        }
        self.memory_peak = 0
        self.process = psutil.Process(os.getpid())
        self.metrics = {}
        
    def start_module(self, module_name):
        """모듈 실행 시작"""
        self.start_time = time.time()
        
    def end_module(self, module_name):
        """모듈 실행 종료 및 런타임 기록"""
        if self.start_time:
            runtime = time.time() - self.start_time
            self.runtimes[module_name] = runtime
            self.start_time = None
            
            # 메모리 사용량 업데이트
            mem_info = self.process.memory_info()
            current_mem = mem_info.rss / 1024 / 1024  # MB
            self.memory_peak = max(self.memory_peak, current_mem)
    
    def add_api_calls(self, module_name, count):
        """API 호출 횟수 추가"""
        if module_name in self.api_calls:
            self.api_calls[module_name] += count
    
    def calculate_metrics(self, maf_path, oncokb_df, pubmed_df, txgnn_df, 
                         clinical_df, merged_df):
        """
        최종 메트릭 계산
        """
        # MAF 변이 개수
        maf = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)
        n_variants = len(maf)
        
        # 각 소스별 약물 수
        #n_oncokb = oncokb_df["Drugs"].dropna().str.split(",").explode().nunique() if "Drugs" in oncokb_df.columns else 0
        n_oncokb = len(oncokb_df) if not oncokb_df.empty else 0
        n_pubmed = len(pubmed_df) if not pubmed_df.empty else 0
        n_txgnn = len(txgnn_df) if not txgnn_df.empty else 0
        n_clinical = clinical_df["drug"].nunique() if not clinical_df.empty else 0
        n_merged = len(merged_df) if not merged_df.empty else 0
        
        # TxGNN Drug Category 분포
        txgnn_categories = {}
        if not txgnn_df.empty and "category" in txgnn_df.columns:
            category_counts = txgnn_df["category"].value_counts().to_dict()
            txgnn_categories = {
                "repurposing": category_counts.get("🔄 Repurposing Priority", 0),
                "current_indication": category_counts.get("✓ Current Indication", 0),
                "investigational": sum(v for k, v in category_counts.items() 
                                      if "Investigational" in k or "🆕" in k),
                "other": sum(v for k, v in category_counts.items() 
                           if k not in ["🔄 Repurposing Priority", "✓ Current Indication"] 
                           and "Investigational" not in k and "🆕" not in k)
            }
        
        # Actionable variants (OncoKB Level 1-4)
        actionable_levels = ["1", "2", "3A", "3B", "4"]
        n_actionable = 0
        if "Level" in oncokb_df.columns:
            n_actionable = oncokb_df[oncokb_df["Level"].isin(actionable_levels)].shape[0]
        
        # Source overlap 계산
        source_counts = Counter()
        if not merged_df.empty and "source" in merged_df.columns:
            for sources in merged_df["source"].dropna():
                n_sources = len(sources.split("|"))
                source_counts[n_sources] += 1
        
        # 총 런타임
        total_runtime = sum(self.runtimes.values())
        
        # 총 API 호출
        total_api_calls = sum(self.api_calls.values())
        
        self.metrics = {
            "patient_id": self.patient_id,
            "n_variants": n_variants,
            "n_actionable_variants": n_actionable,
            
            # Runtime (초)
            "runtime_oncokb": self.runtimes.get("oncokb", 0),
            "runtime_pubmed": self.runtimes.get("pubmed", 0),
            "runtime_txgnn": self.runtimes.get("txgnn", 0),
            "runtime_clinical": self.runtimes.get("clinical", 0),
            "runtime_total": total_runtime,
            
            # API Calls
            "api_calls_oncokb": self.api_calls["oncokb"],
            "api_calls_pubmed": self.api_calls["pubmed"],
            "api_calls_clinical": self.api_calls["clinicaltrials"],
            "api_calls_total": total_api_calls,
            
            # Memory
            "memory_peak_mb": round(self.memory_peak, 2),
            
            # Drug counts
            "n_drugs_oncokb": n_oncokb,
            "n_drugs_pubmed": n_pubmed,
            "n_drugs_txgnn": n_txgnn,
            "n_drugs_clinical": n_clinical,
            "n_drugs_merged": n_merged,
            
            # TxGNN categories
            "txgnn_repurposing": txgnn_categories.get("repurposing", 0),
            "txgnn_current_indication": txgnn_categories.get("current_indication", 0),
            "txgnn_investigational": txgnn_categories.get("investigational", 0),
            "txgnn_other": txgnn_categories.get("other", 0),
            
            # Overlap
            "drugs_single_source": source_counts.get(1, 0),
            "drugs_two_sources": source_counts.get(2, 0),
            "drugs_three_sources": source_counts.get(3, 0),
            "drugs_all_sources": source_counts.get(4, 0),
            
            # Score statistics
            "combined_score_mean": merged_df["combined_score"].mean() if not merged_df.empty else 0,
            "combined_score_median": merged_df["combined_score"].median() if not merged_df.empty else 0,
            "combined_score_max": merged_df["combined_score"].max() if not merged_df.empty else 0,
        }
        
        return self.metrics
    
    def to_dataframe(self):
        """메트릭을 DataFrame으로 변환"""
        if not self.metrics:
            return pd.DataFrame()
        
        # 보기 좋게 카테고리별로 정리
        data = []
        
        # Basic Info
        data.append({"Category": "Basic Info", "Metric": "Patient ID", "Value": self.metrics["patient_id"]})
        data.append({"Category": "Basic Info", "Metric": "Total Variants", "Value": self.metrics["n_variants"]})
        data.append({"Category": "Basic Info", "Metric": "Actionable Variants", "Value": self.metrics["n_actionable_variants"]})
        
        # Runtime
        data.append({"Category": "Runtime (sec)", "Metric": "OncoKB", "Value": f"{self.metrics['runtime_oncokb']:.2f}"})
        data.append({"Category": "Runtime (sec)", "Metric": "PubMed", "Value": f"{self.metrics['runtime_pubmed']:.2f}"})
        data.append({"Category": "Runtime (sec)", "Metric": "TxGNN", "Value": f"{self.metrics['runtime_txgnn']:.2f}"})
        data.append({"Category": "Runtime (sec)", "Metric": "ClinicalTrials", "Value": f"{self.metrics['runtime_clinical']:.2f}"})
        data.append({"Category": "Runtime (sec)", "Metric": "Total", "Value": f"{self.metrics['runtime_total']:.2f}"})
        
        # API Calls
        data.append({"Category": "API Calls", "Metric": "OncoKB", "Value": self.metrics["api_calls_oncokb"]})
        data.append({"Category": "API Calls", "Metric": "PubMed", "Value": self.metrics["api_calls_pubmed"]})
        data.append({"Category": "API Calls", "Metric": "ClinicalTrials", "Value": self.metrics["api_calls_clinical"]})
        data.append({"Category": "API Calls", "Metric": "Total", "Value": self.metrics["api_calls_total"]})
        
        # Memory
        data.append({"Category": "Memory", "Metric": "Peak Usage (MB)", "Value": self.metrics["memory_peak_mb"]})
        
        # Drug Counts
        data.append({"Category": "Drug Counts", "Metric": "OncoKB", "Value": self.metrics["n_drugs_oncokb"]})
        data.append({"Category": "Drug Counts", "Metric": "PubMed", "Value": self.metrics["n_drugs_pubmed"]})
        data.append({"Category": "Drug Counts", "Metric": "TxGNN", "Value": self.metrics["n_drugs_txgnn"]})
        data.append({"Category": "Drug Counts", "Metric": "ClinicalTrials", "Value": self.metrics["n_drugs_clinical"]})
        data.append({"Category": "Drug Counts", "Metric": "Merged (Total)", "Value": self.metrics["n_drugs_merged"]})
        
        # TxGNN Categories
        data.append({"Category": "TxGNN Categories", "Metric": "Repurposing", "Value": self.metrics["txgnn_repurposing"]})
        data.append({"Category": "TxGNN Categories", "Metric": "Current Indication", "Value": self.metrics["txgnn_current_indication"]})
        data.append({"Category": "TxGNN Categories", "Metric": "Investigational", "Value": self.metrics["txgnn_investigational"]})
        data.append({"Category": "TxGNN Categories", "Metric": "Other", "Value": self.metrics["txgnn_other"]})
        
        # Source Overlap
        data.append({"Category": "Source Overlap", "Metric": "Single Source", "Value": self.metrics["drugs_single_source"]})
        data.append({"Category": "Source Overlap", "Metric": "Two Sources", "Value": self.metrics["drugs_two_sources"]})
        data.append({"Category": "Source Overlap", "Metric": "Three Sources", "Value": self.metrics["drugs_three_sources"]})
        data.append({"Category": "Source Overlap", "Metric": "All Sources", "Value": self.metrics["drugs_all_sources"]})
        
        # Score Stats
        data.append({"Category": "Score Statistics", "Metric": "Mean", "Value": f"{self.metrics['combined_score_mean']:.3f}"})
        data.append({"Category": "Score Statistics", "Metric": "Median", "Value": f"{self.metrics['combined_score_median']:.3f}"})
        data.append({"Category": "Score Statistics", "Metric": "Max", "Value": f"{self.metrics['combined_score_max']:.3f}"})
        
        return pd.DataFrame(data)