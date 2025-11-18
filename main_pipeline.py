import argparse
import pandas as pd
from pathlib import Path

# === Module import ===
from pubmed_module import run_pubmed
from txgnn_module import run_txgnn
from oncokb_module import run_oncokb_and_extract
from clinicaltrials_module import run_clinical_trials, build_trial_summary_table
from performance_tracker import PerformanceTracker

# PDF related
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
import matplotlib.pyplot as plt
import networkx as nx
import os

def generate_barplot_top_drugs(merged_df, outdir):
    """Generation of top20 drugs barplot"""
    top_df = merged_df.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["drug"], top_df["combined_score"])
    plt.xlabel("Combined Score")
    plt.ylabel("Drug")
    plt.title("Top 20 Recommended Drugs")
    plt.gca().invert_yaxis()
    outpath = os.path.join(outdir, "top20_drugs.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath

def generate_txgnn_graph_plot(disease, mutation_genes, txgnn_df, outdir):
    """TxGNN network graph"""
    G = nx.Graph()
    G.add_node(disease, color="red", size=800)
    
    for g in mutation_genes:
        G.add_node(g, color="blue", size=600)
        G.add_edge(disease, g)
    
    for _, row in txgnn_df.iterrows():
        drug = row["drug"]
        genes = row["connected_genes"].split(",") if "connected_genes" in row and isinstance(row["connected_genes"], str) else []
        G.add_node(drug, color="green", size=500)
        for g in genes:
            g = g.strip()
            if g:
                G.add_edge(g, drug)
    
    pos = nx.spring_layout(G, k=0.6)
    plt.figure(figsize=(10, 8))
    
    colors = []
    sizes = []
    for n in G.nodes():
        node_data = G.nodes[n]
        colors.append(node_data.get("color", "gray"))
        sizes.append(node_data.get("size", 400))
    
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=sizes, 
            font_size=8, edge_color="gray")
    
    outpath = os.path.join(outdir, "txgnn_graph.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath

def generate_pdf_report(summary_df, oncokb_df, pubmed_df, txgnn_df, clinical_df,
                       merged_df, maf_path, cancer_type, output_pdf_path, output_dir):
    """Generation of PDF report"""
    styles = getSampleStyleSheet()
    styleH = styles["Heading1"]
    styleN = styles["BodyText"]
    
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    story = []
    
    story.append(Paragraph("Precision Oncology Report", styleH))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Cancer Type: {cancer_type}", styleN))
    story.append(Paragraph(f"Input MAF: {maf_path}", styleN))
    story.append(Spacer(1, 24))
    story.append(PageBreak())
    
    story.append(Paragraph("1. Summary", styleH))
    story.append(Spacer(1, 12))
    
    tbl_data = [["Key", "Value"]]
    for _, row in summary_df.iterrows():
        tbl_data.append([row["Key"], row["Value"]])
    
    table = Table(tbl_data, colWidths=[120, 350])
    table.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 1, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(table)
    story.append(Spacer(1, 18))
    story.append(PageBreak())
    
    story.append(Paragraph("2. Top 20 Recommended Drugs", styleH))
    story.append(Spacer(1, 12))
    barplot_path = generate_barplot_top_drugs(merged_df, output_dir)
    story.append(Image(barplot_path, width=480, height=320))
    story.append(PageBreak())
    
    story.append(Paragraph("3. TxGNN Graph-Based Drug Relationships", styleH))
    story.append(Spacer(1, 12))
    maf = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)
    mutation_genes = sorted(maf["Hugo_Symbol"].dropna().unique().tolist())
    graph_path = generate_txgnn_graph_plot(cancer_type, mutation_genes, txgnn_df, output_dir)
    story.append(Image(graph_path, width=480, height=360))
    story.append(PageBreak())
    
    story.append(Paragraph("4. ClinicalTrials Evidence Summary", styleH))
    story.append(Spacer(1, 12))
    ct_tbl = [["Drug", "#Trials", "Top NCT", "Phase"]]
    for _, row in clinical_df.head(30).iterrows():
        ct_tbl.append([
            row.get("drug", ""),
            row.get("n_ct", row.get("n_clinical_trials", "")),
            row.get("top_nct_id", ""),
            row.get("top_phase", "")
        ])
    
    table2 = Table(ct_tbl, colWidths=[120, 80, 120, 80])
    table2.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 1, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(table2)
    
    doc.build(story)
    print(f"📄 PDF saved → {output_pdf_path}")

DISEASE_MAP = {
    "NSCLC": "lung cancer",
    "LUAD": "lung adenocarcinoma",
    "LUSC": "lung squamous cell carcinoma",
    "BRCA": "breast cancer",
    "CRC": "colon cancer",
    "COAD": "colon cancer",
    "GBM": "glioblastoma",
    "SKCM": "melanoma",
    "OV": "ovarian cancer",
    "HNSC": "head and neck cancer",
}

def normalize_cancer_type(cancer_type: str) -> str:
    key = cancer_type.upper().replace(" ", "")
    return DISEASE_MAP.get(key, cancer_type)

def normalize_for_oncokb(cancer_type: str) -> str:
    mapping = {
        "NSCLC": "Non-Small Cell Lung Cancer",
        "LUAD": "Lung Adenocarcinoma",
        "LUSC": "Lung Squamous Cell Carcinoma",
        "GBM": "Glioblastoma Multiforme",
        "GLIOBLASTOMA": "Glioblastoma Multiforme",
        "BRCA": "Breast Invasive Carcinoma",
        "BREAST CANCER": "Breast Invasive Carcinoma",
        "PDAC": "Pancreatic Adenocarcinoma",
        "PANCREATIC CANCER": "Pancreatic Adenocarcinoma",
        "COAD": "Colon Adenocarcinoma",
        "CRC": "Colon Adenocarcinoma",
        "COLON CANCER": "Colon Adenocarcinoma"
    }

    key = cancer_type.strip().upper()
    return mapping.get(key, cancer_type)

def merge_all_results(oncokb_df, pubmed_df, txgnn_df, clinical_summary_df):
    """PubMed + TxGNN + OncoKB + ClinicalTrials summary 병합"""
    
    def norm(s):
        return s.fillna("").str.upper().str.strip()
    
    if not pubmed_df.empty:
        pubmed_df["drug_norm"] = norm(pubmed_df["drug"])
    else:
        pubmed_df["drug_norm"] = ""
    
    if not txgnn_df.empty:
        txgnn_df["drug_norm"] = norm(txgnn_df["drug"])
    else:
        txgnn_df["drug_norm"] = ""
    
    if "Drugs" in oncokb_df.columns:
        oncokb_ex = oncokb_df.copy()
        oncokb_ex["Drugs"] = oncokb_ex["Drugs"].fillna("")
        oncokb_ex = oncokb_ex[oncokb_ex["Drugs"] != ""]
        oncokb_ex = oncokb_ex.assign(Drug=oncokb_ex["Drugs"].str.split(","))
        oncokb_ex = oncokb_ex.explode("Drug")
        oncokb_ex["drug_norm"] = norm(oncokb_ex["Drug"])
    else:
        oncokb_ex = oncokb_df.copy()
        oncokb_ex["drug_norm"] = ""
    
    merged = pd.merge(pubmed_df, txgnn_df, on="drug_norm", how="outer", suffixes=("_pubmed", "_txgnn"))
    merged = pd.merge(merged, oncokb_ex, on="drug_norm", how="outer", suffixes=("", "_oncokb"))
    
    def choose_name(row):
        for col in ["drug_pubmed", "drug_txgnn", "drug"]:
            if col in row and pd.notna(row[col]) and row[col] != "":
                return row[col]
        return row.get("drug_norm", "UNKNOWN")
    
    merged["drug"] = merged.apply(choose_name, axis=1)
    
    level_score_map = {"1": 10.0, "2": 8.0, "3A": 6.0, "3B": 4.0, "4": 2.0, "R1": -5.0, "R2": -3.0}
    
    def get_oncokb_score(row):
        level = row.get("oncokb_level", "")
        if pd.isna(level) or level == "":
            return 0.0
        return level_score_map.get(str(level), 0.0)
    
    merged["oncokb_score"] = merged.apply(get_oncokb_score, axis=1)
    
    txgnn_score = merged["txgnn_score"].fillna(0) if "txgnn_score" in merged.columns else 0
    pubmed_score = merged["mention_count"].fillna(0) if "mention_count" in merged.columns else 0
    oncokb_score = merged["oncokb_score"].fillna(0)
    
    merged["combined_score"] = txgnn_score * 0.4 + pubmed_score * 0.3 + oncokb_score * 0.3
    
    merged["source"] = ""
    if "txgnn_score" in merged.columns:
        merged.loc[merged["txgnn_score"].notna(), "source"] += "TxGNN|"
    if "mention_count" in merged.columns:
        merged.loc[merged["mention_count"].notna(), "source"] += "PubMed|"
    if "oncokb_level" in merged.columns:
        merged.loc[merged["oncokb_level"].notna(), "source"] += "OncoKB|"
    
    merged["source"] = merged["source"].str.rstrip("|")
    merged = merged.sort_values("combined_score", ascending=False).reset_index(drop=True)
    
    if clinical_summary_df is not None and not clinical_summary_df.empty:
        clinical_summary_df["drug_norm"] = norm(clinical_summary_df["drug"])
        merged = merged.merge(
            clinical_summary_df[["drug_norm", "n_clinical_trials", "top_nct_id", "top_phase", "top_title"]],
            on="drug_norm",
            how="left"
        )
    
    merged = merged.sort_values("combined_score", ascending=False)
    return merged

def build_summary_sheet(merged_df, maf_path, cancer_type):
    maf = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)
    n_variants = len(maf)
    
    rows = []
    rows.append({"Key": "Cancer Type", "Value": cancer_type})
    rows.append({"Key": "MAF Variant Count", "Value": n_variants})
    
    for i, row in merged_df.head(20).iterrows():
        rows.append({
            "Key": f"TopDrug_{i+1}",
            "Value": f"{row.get('drug', 'NA')} (score={row.get('combined_score', 0):.3f})"
        })
    
    return pd.DataFrame(rows)

def run_pipeline(maf_path, cancer_type, oncokb_token, annotator_path, pubmed_token,
                 data_folder, txgnn_root, output_dir, patient_id=None):
    
    maf_path = Path(maf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Performance Tracker initialization
    if patient_id is None:
        patient_id = maf_path.stem  
    tracker = PerformanceTracker(patient_id=patient_id)
    
    cancer_type_raw = cancer_type
    cancer_type_tx = normalize_cancer_type(cancer_type)
    
    print(f"🧬 입력 암종 = {cancer_type_raw}")
    print(f"🔧 TxGNN friendly disease name = {cancer_type_tx}")
    
    oncokb_out = output_dir / "oncokb_output.tsv"
    pubmed_out = output_dir / "pubmed_output.tsv"
    txgnn_out = output_dir / "txgnn_output.tsv"
    clinical_out = output_dir / "clinicaltrials_output.tsv"
    final_report = output_dir / "final_report.xlsx"
    
    # ---------------------------------------------------------
    print("\n==========================")
    print("1) Running OncoKB annotator")
    print("==========================")
    tracker.start_module("oncokb")
    cancer_type_oncokb = normalize_for_oncokb(cancer_type_raw)
    oncokb_df = run_oncokb_and_extract(
        maf_path=str(maf_path),
        cancer_type=cancer_type_oncokb,
        output_tsv_path=str(oncokb_out),
        annotator_path=annotator_path,
        oncokb_api_token=oncokb_token
    )
    tracker.end_module("oncokb")
    tracker.add_api_calls("oncokb", 1)  
    
    # ---------------------------------------------------------
    print("\n==========================")
    print("2) Running PubMed analysis")
    print("==========================")
    tracker.start_module("pubmed")
    pubmed_df = run_pubmed(
        maf_path=str(maf_path),
        cancer_type=cancer_type_raw,
        output_path=str(pubmed_out),
        pubmed_token=pubmed_token
    )
    tracker.end_module("pubmed")

    if not pubmed_df.empty:
        tracker.add_api_calls("pubmed", len(pubmed_df))
    
    # ---------------------------------------------------------
    print("\n==========================")
    print("3) Running TxGNN")
    print("==========================")
    tracker.start_module("txgnn")
    txgnn_df = run_txgnn(
        maf_path=str(maf_path),
        cancer_type=cancer_type_tx,
        output_path=str(txgnn_out),
        data_folder=data_folder,
        txgnn_root=txgnn_root
    )
    tracker.end_module("txgnn")
    
    # ---------------------------------------------------------
    print("\n==========================")
    print("4) Searching ClinicalTrials.gov")
    print("==========================")
    
    all_drugs = set()
    if not pubmed_df.empty:
        all_drugs.update(pubmed_df["drug"].dropna().str.upper().tolist())
    
    if not txgnn_df.empty:
        filtered_txgnn = txgnn_df[txgnn_df.get("current_use", "") != "✓"]
        all_drugs.update(filtered_txgnn["drug"].dropna().str.upper().tolist())
        print(f"   Excluding Current Indication: {len(txgnn_df) - len(filtered_txgnn)}")
    
    all_drugs = list(all_drugs)
    print(f"   ✓ ClinicalTrials search candidates: {len(all_drugs)} drugs")
    
    tracker.start_module("clinical")
    clinical_df = run_clinical_trials(
        drug_list=all_drugs,
        cancer_type=cancer_type_tx,
        output_path=str(clinical_out)
    )
    tracker.end_module("clinical")
    tracker.add_api_calls("clinicaltrials", len(all_drugs))
    
    clinical_summary_df = build_trial_summary_table(clinical_df)
    
    # ---------------------------------------------------------
    print("\n==========================")
    print("5) Merging all results")
    print("==========================")
    merged_df = merge_all_results(oncokb_df, pubmed_df, txgnn_df, clinical_summary_df)
    
    # Performance metric
    tracker.calculate_metrics(
        maf_path=str(maf_path),
        oncokb_df=oncokb_df,
        pubmed_df=pubmed_df,
        txgnn_df=txgnn_df,
        clinical_df=clinical_df,
        merged_df=merged_df
    )
    
    performance_df = tracker.to_dataframe()
    
    # ---------------------------------------------------------
    print("\n==========================")
    print("6) Generating Excel Report")
    print("==========================")
    
    summary_df = build_summary_sheet(merged_df, maf_path, cancer_type_raw)
    
    with pd.ExcelWriter(final_report, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        performance_df.to_excel(writer, sheet_name="Performance", index=False)  # ✅ 추가!
        oncokb_df.to_excel(writer, sheet_name="OncoKB", index=False)
        pubmed_df.to_excel(writer, sheet_name="PubMed", index=False)
        txgnn_df.to_excel(writer, sheet_name="TxGNN", index=False)
        clinical_df.to_excel(writer, sheet_name="ClinicalTrials", index=False)
        merged_df.to_excel(writer, sheet_name="Merged_Drugs", index=False)
    
    print(f"\n🎉 Final report saved to: {final_report}\n")
    
    # ---------------------------------------------------------
    print("\n==========================")
    print("7) Generating PDF Report")
    print("==========================")
    
    pdf_path = output_dir / "final_report.pdf"
    generate_pdf_report(
        summary_df, oncokb_df, pubmed_df, txgnn_df, clinical_summary_df,
        merged_df, maf_path, cancer_type_raw, str(pdf_path), str(output_dir)
    )
    
 
    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total Runtime: {tracker.metrics['runtime_total']:.2f} sec")
    print(f"Total API Calls: {tracker.metrics['api_calls_total']}")
    print(f"Peak Memory: {tracker.metrics['memory_peak_mb']:.2f} MB")
    print(f"Total Drugs: {tracker.metrics['n_drugs_merged']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--maf", required=True, help="Input MAF file")
    parser.add_argument("--cancer", required=True, help="Cancer type (e.g., NSCLC, LUAD)")
    parser.add_argument("--oncokb_token", required=True, help="OncoKB API token")
    parser.add_argument("--annotator", required=True, help="Path to MafAnnotator.py")
    parser.add_argument("--pubmed_token", required=True, help="Pubmed API token")
    parser.add_argument("--txgnn_data", required=True, help="TxGNN data folder")
    parser.add_argument("--txgnn_root", required=True, help="TxGNN root folder")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--patient_id", default=None, help="Patient ID (optional)")
    
    args = parser.parse_args()
    
    run_pipeline(
        maf_path=args.maf,
        cancer_type=args.cancer,
        oncokb_token=args.oncokb_token,
        annotator_path=args.annotator,
        pubmed_token=args.pubmed_token,
        data_folder=args.txgnn_data,
        txgnn_root=args.txgnn_root,
        output_dir=args.outdir,
        patient_id=args.patient_id
    )