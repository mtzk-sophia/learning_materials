"""
Step 1: データ読み込み・前処理
================================
- financial_data.csvの読み込みと加工
- 有価証券報告書（PDF）からのテキスト抽出
- 企業別マスタデータの作成
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
import google.generativeai as genai


# =============================================================================
# 設定
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 出力ディレクトリの作成
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. financial_data.csv の読み込み・前処理
# =============================================================================
def load_financial_data() -> pd.DataFrame:
    """財務データCSVを読み込む"""
    csv_path = RAW_DIR / "financial_data.csv"
    df = pd.read_csv(csv_path)
    print(f"[INFO] 財務データ読み込み完了: {len(df)}行")
    return df


def preprocess_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """財務データの前処理を行う"""
    df = df.copy()
    
    # カラム名の整理（空白除去）
    df.columns = df.columns.str.strip()
    
    # 企業コードを文字列として扱う
    df["コード"] = df["コード"].astype(str)
    
    # 年度を整数に
    df["YEAR"] = df["YEAR"].astype(int)
    
    # 数値列の特定（コード、本社所在地、市場・商品区分、業種分類、YEAR以外）
    non_numeric_cols = ["コード", "本社所在地", "市場・商品区分", "業種分類", "YEAR"]
    numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
    
    # 従業員数と資本金以外は円単位 → 億円に変換
    yen_cols = [col for col in numeric_cols if col not in ["従業員数（連結）", "資本金（億円）"]]
    
    # 億円変換（1億 = 100,000,000円）
    for col in yen_cols:
        if col in df.columns:
            df[f"{col}_億円"] = df[col] / 100_000_000
    
    print(f"[INFO] 財務データ前処理完了: {len(df)}行, {len(df.columns)}列")
    return df


def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """財務指標を計算して追加"""
    df = df.copy()
    
    # 収益性指標
    df["売上高営業利益率"] = df["営業利益"] / df["売上高"] * 100
    df["売上高経常利益率"] = df["経常利益"] / df["売上高"] * 100
    df["売上高当期純利益率"] = df["当期純利益"] / df["売上高"] * 100
    df["ROA"] = df["当期純利益"] / df["総資産"] * 100
    df["ROE"] = df["当期純利益"] / df["純資産"] * 100
    
    # 安全性指標
    df["自己資本比率"] = df["純資産"] / df["総資産"] * 100
    df["流動比率"] = df["流動資産"] / df["流動負債"] * 100
    df["固定比率"] = df["固定資産"] / df["純資産"] * 100
    
    # 負債 = 総資産 - 純資産
    df["負債合計"] = df["総資産"] - df["純資産"]
    df["負債比率"] = df["負債合計"] / df["純資産"] * 100
    
    # 効率性指標
    df["総資産回転率"] = df["売上高"] / df["総資産"]
    
    # 建設業特有指標
    if "売上総利益_完成工事総利益" in df.columns and "売上高_完成工事高" in df.columns:
        df["完成工事総利益率"] = df["売上総利益_完成工事総利益"] / df["売上高_完成工事高"] * 100
    
    # 従業員一人当たり売上高（百万円）
    df["従業員一人当たり売上高_百万円"] = df["売上高"] / df["従業員数（連結）"] / 1_000_000
    
    # キャッシュフロー指標
    df["フリーキャッシュフロー"] = df["営業活動によるキャッシュ・フロー"] + df["投資活動によるキャッシュ・フロー"]
    
    print("[INFO] 財務指標計算完了")
    return df


def calculate_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """前年比成長率を計算"""
    df = df.copy()
    df = df.sort_values(["コード", "YEAR"])
    
    # 成長率を計算する項目
    growth_items = ["売上高", "営業利益", "経常利益", "当期純利益", "総資産", "純資産"]
    
    for item in growth_items:
        if item in df.columns:
            df[f"{item}_前年比"] = df.groupby("コード")[item].pct_change() * 100
    
    print("[INFO] 成長率計算完了")
    return df


def get_company_summary(df: pd.DataFrame) -> pd.DataFrame:
    """企業別サマリーを作成（最新年度のデータ）"""
    latest_year = df["YEAR"].max()
    summary = df[df["YEAR"] == latest_year].copy()
    
    # 必要な列を選択
    summary_cols = [
        "コード", "本社所在地", "市場・商品区分", "従業員数（連結）", 
        "資本金（億円）", "業種分類",
        "売上高_億円", "営業利益_億円", "経常利益_億円", "当期純利益_億円",
        "総資産_億円", "純資産_億円",
        "売上高営業利益率", "売上高経常利益率", "ROE", "ROA",
        "自己資本比率", "流動比率"
    ]
    
    # 存在する列のみ選択
    available_cols = [col for col in summary_cols if col in summary.columns]
    summary = summary[available_cols].reset_index(drop=True)
    
    return summary


# =============================================================================
# 2. 有価証券報告書（PDF）のテキスト抽出（Gemini File API使用）
# =============================================================================
def extract_text_from_pdf_with_gemini(pdf_path: Path, api_key: str) -> str:
    """GeminiのFile APIを使ってPDFをMarkdownに変換"""
    print(f"    Gemini APIでPDF処理中: {pdf_path.name}")
    
    # Gemini設定
    genai.configure(api_key=api_key)
    
    # PDFファイルをアップロード
    uploaded_file = genai.upload_file(path=str(pdf_path))
    
    # Markdownに変換
    model = genai.GenerativeModel("gemini-2.5-pro")  # Flash で十分
    prompt = """この有価証券報告書PDFの内容を、構造を保ったままMarkdown形式に変換してください。

【要求】
- 見出し構造を保持（# ## ### を使用）
- 表はMarkdownテーブル形式に変換
- 本文は段落単位で整理
- ヘッダー・フッター・ページ番号は除外
- 「架空・サンプルデータ」等の注記は残す

全文を出力してください。"""
    
    response = model.generate_content([uploaded_file, prompt])
    
    # アップロードファイルを削除（クリーンアップ）
    genai.delete_file(uploaded_file.name)
    
    return response.text


def extract_company_code_from_filename(filename: str) -> Optional[str]:
    """ファイル名から企業コードを抽出"""
    match = re.search(r"（(\d+)）", filename)
    if match:
        return match.group(1)
    return None


def markdown_to_structured_json(markdown_text: str, company_code: str, api_key: str) -> dict:
    """Markdownテキストから構造化されたJSONを生成（Gemini使用）"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""以下のMarkdown形式の有価証券報告書から、重要な情報を抽出してJSON形式で出力してください。

【抽出項目と記述のポイント】
1. 事業内容: セグメント構成、各事業の説明、官公庁/民間・元請/下請の販路構成
2. 経営方針: 経営理念、中期経営計画の名称・期間・重点テーマ、基本方針
3. 中期経営計画の目標数値: 売上高・営業利益・営業利益率・ROE等の現在値と目標値
4. セグメント別業績: 各セグメントの売上高・利益、前年比較
5. 受注・販売実績: 受注高、受注残高、完成工事高（建設業固有の重要指標）
6. 経営者の分析: 増減益の要因分析、経営者自身の現状認識と見通し
7. リスク情報: 主要リスクとその対応策（リスクマトリクスがあればリスク値も）
8. 経営上の課題: 対処すべき課題の一覧と具体的対処方針
9. 研究開発: 研究開発活動の内容、費用、成果
10. 設備投資（実績）: 当期の設備投資額、セグメント別内訳
11. 設備投資（計画）: 翌期以降の設備投資計画（項目、金額、時期、目的）
12. 従業員の状況: 従業員数、平均年齢、セグメント別人数、女性管理職比率等
13. サステナビリティ: TCFD対応、ESG取組、環境目標（CO2削減目標等）
14. DX_GX: DX/GXに関する具体的な取組・技術・投資（ICT施工、BIM/CIM、ZEB等）

【出力形式】
JSON形式で出力してください。各項目はテキストとして格納してください。
```json
{{
  "事業内容": "...",
  "経営方針": "...",
  "中期経営計画_目標数値": "...",
  "セグメント別業績": "...",
  "受注販売実績": "...",
  "経営者の分析": "...",
  "リスク情報": "...",
  "経営上の課題": "...",
  "研究開発": "...",
  "設備投資_実績": "...",
  "設備投資_計画": "...",
  "従業員の状況": "...",
  "サステナビリティ": "...",
  "DX_GX": "..."
}}
```

【Markdownテキスト】
{markdown_text[:30000]}
"""
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text
        
        # JSONを抽出
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        return json.loads(result_text.strip())
    except Exception as e:
        print(f"[WARNING] JSON抽出エラー（{company_code}）: {e}")
        # フォールバック: 正規表現で分割
        return parse_securities_report_from_markdown(markdown_text)


def parse_securities_report_from_markdown(markdown_text: str) -> dict:
    """Markdown形式の有価証券報告書をセクション別に分割"""
    sections = {
        "事業内容": "",
        "経営方針": "",
        "リスク情報": "",
        "経営上の課題": "",
        "研究開発": "",
        "設備投資": "",
        "従業員の状況": "",
        "サステナビリティ": "",
        "その他": ""
    }
    
    # セクションを特定するためのキーワードパターン
    section_patterns = {
        "事業内容": r"(事業の内容|主要な事業の内容|【事業の内容】)",
        "経営方針": r"(経営方針|経営環境及び対処すべき課題|経営上の目標|経営戦略)",
        "リスク情報": r"(事業等のリスク|リスク情報|【事業等のリスク】)",
        "経営上の課題": r"(対処すべき課題|経営上の重要な課題)",
        "研究開発": r"研究開発活動",
        "設備投資": r"(設備投資|設備の状況)",
        "従業員の状況": r"従業員の状況",
        "サステナビリティ": r"サステナビリティ"
    }
    
    current_section = "その他"
    lines = markdown_text.split("\n")
    
    for line in lines:
        # Markdownの見出し（# ## ###）も考慮
        line_stripped = line.strip().lstrip("#").strip()
        
        # セクションの判定
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, line_stripped):
                current_section = section_name
                break
        
        sections[current_section] += line + "\n"
    
    return sections


def load_all_securities_reports(api_key: Optional[str] = None) -> dict:
    """全ての有価証券報告書を読み込む（Gemini使用）"""
    # API キー取得
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[WARNING] GOOGLE_API_KEY が設定されていません。")
        print("          PDFからのテキスト抽出をスキップします。")
        print("          環境変数を設定して再実行してください: export GOOGLE_API_KEY='your-key'")
        return {}
    
    reports_dir = RAW_DIR / "securities_report"
    reports = {}
    
    for pdf_file in reports_dir.glob("*.pdf"):
        company_code = extract_company_code_from_filename(pdf_file.name)
        if company_code:
            print(f"[INFO] PDF→Markdown変換中: {pdf_file.name}")
            try:
                # Step 1: PDF → Markdown
                markdown_text = extract_text_from_pdf_with_gemini(pdf_file, api_key)
                
                # Step 2: Markdown → 構造化JSON
                print(f"    Markdown→JSON構造化中...")
                import time
                time.sleep(2)
                sections = markdown_to_structured_json(markdown_text, company_code, api_key)
                
                reports[company_code] = {
                    "filename": pdf_file.name,
                    "raw_text": markdown_text,  # Markdown形式
                    "sections": sections,
                    "text_length": len(markdown_text)
                }
                
                # API制限対策
                time.sleep(3)
                
            except Exception as e:
                print(f"[ERROR] {pdf_file.name} の処理でエラー: {e}")
                continue
    
    print(f"[INFO] 有価証券報告書読み込み完了: {len(reports)}社")
    return reports


# =============================================================================
# 3. 企業マスタデータの作成
# =============================================================================
@dataclass
class CompanyMaster:
    """企業マスタ情報"""
    code: str
    location: str
    market: str
    employees: int
    capital: float  # 億円
    industry: str
    years: list  # データが存在する年度リスト
    
    # 最新年度の主要財務データ
    latest_revenue: float  # 売上高（億円）
    latest_operating_income: float  # 営業利益（億円）
    latest_net_income: float  # 当期純利益（億円）
    latest_total_assets: float  # 総資産（億円）
    latest_equity: float  # 純資産（億円）
    
    # 有価証券報告書情報
    has_securities_report: bool
    securities_report_length: Optional[int]


def create_company_master(df: pd.DataFrame, reports: dict) -> list[CompanyMaster]:
    """企業マスタを作成"""
    masters = []
    
    for code in df["コード"].unique():
        company_data = df[df["コード"] == code]
        latest = company_data[company_data["YEAR"] == company_data["YEAR"].max()].iloc[0]
        
        has_report = code in reports
        report_length = reports[code]["text_length"] if has_report else None
        
        master = CompanyMaster(
            code=code,
            location=latest["本社所在地"],
            market=latest["市場・商品区分"],
            employees=int(latest["従業員数（連結）"]),
            capital=float(latest["資本金（億円）"]),
            industry=latest["業種分類"],
            years=sorted(company_data["YEAR"].tolist()),
            latest_revenue=float(latest["売上高"]) / 100_000_000,
            latest_operating_income=float(latest["営業利益"]) / 100_000_000,
            latest_net_income=float(latest["当期純利益"]) / 100_000_000,
            latest_total_assets=float(latest["総資産"]) / 100_000_000,
            latest_equity=float(latest["純資産"]) / 100_000_000,
            has_securities_report=has_report,
            securities_report_length=report_length
        )
        masters.append(master)
    
    print(f"[INFO] 企業マスタ作成完了: {len(masters)}社")
    return masters


# =============================================================================
# 4. データ出力
# =============================================================================
def save_processed_data(
    df: pd.DataFrame, 
    summary: pd.DataFrame,
    masters: list[CompanyMaster],
    reports: dict
):
    """前処理済みデータを保存"""
    
    # 財務データ（全体）
    df.to_csv(PROCESSED_DIR / "financial_data_processed.csv", index=False, encoding="utf-8-sig")
    print(f"[SAVE] financial_data_processed.csv")
    
    # 企業サマリー
    summary.to_csv(PROCESSED_DIR / "company_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[SAVE] company_summary.csv")
    
    # 企業マスタ（JSON）
    masters_dict = [asdict(m) for m in masters]
    with open(PROCESSED_DIR / "company_master.json", "w", encoding="utf-8") as f:
        json.dump(masters_dict, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] company_master.json")
    
    # 企業別財務データ
    company_data_dir = PROCESSED_DIR / "by_company"
    company_data_dir.mkdir(exist_ok=True)
    
    for code in df["コード"].unique():
        company_df = df[df["コード"] == code]
        company_df.to_csv(
            company_data_dir / f"{code}_financial.csv",
            index=False, 
            encoding="utf-8-sig"
        )
    print(f"[SAVE] 企業別財務データ: {len(df['コード'].unique())}社")
    
    # 有価証券報告書テキスト（企業別）
    reports_dir = PROCESSED_DIR / "securities_reports_text"
    reports_dir.mkdir(exist_ok=True)
    
    for code, report in reports.items():
        # 全文テキスト
        with open(reports_dir / f"{code}_full_text.txt", "w", encoding="utf-8") as f:
            f.write(report["raw_text"])
        
        # セクション別テキスト（JSON）
        with open(reports_dir / f"{code}_sections.json", "w", encoding="utf-8") as f:
            json.dump(report["sections"], f, ensure_ascii=False, indent=2)
    
    print(f"[SAVE] 有価証券報告書テキスト: {len(reports)}社")


def print_data_overview(df: pd.DataFrame, masters: list[CompanyMaster]):
    """データ概要を表示"""
    print("\n" + "="*60)
    print("データ概要")
    print("="*60)
    
    print(f"\n■ 財務データ")
    print(f"  - レコード数: {len(df)}")
    print(f"  - 企業数: {df['コード'].nunique()}")
    print(f"  - 年度: {sorted(df['YEAR'].unique())}")
    print(f"  - カラム数: {len(df.columns)}")
    
    print(f"\n■ 企業一覧")
    print("-" * 60)
    for m in masters:
        print(f"  {m.code}: {m.location} | {m.industry} | "
              f"売上 {m.latest_revenue:.1f}億円 | "
              f"従業員 {m.employees}名")
    
    print("\n■ 欠損値確認")
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print("  欠損値あり:")
        for col, count in null_cols.items():
            print(f"    - {col}: {count}件")
    else:
        print("  欠損値なし")
    
    print("="*60 + "\n")


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print("="*60)
    print("Step 1: データ読み込み・前処理")
    print("="*60 + "\n")
    
    # 1. 財務データの読み込み・前処理
    print("[Step 1.1] 財務データ処理")
    df = load_financial_data()
    df = preprocess_financial_data(df)
    df = add_financial_ratios(df)
    df = calculate_growth_rates(df)
    
    # 企業サマリーの作成
    summary = get_company_summary(df)
    
    # 2. 有価証券報告書の読み込み（Gemini使用）
    print("\n[Step 1.2] 有価証券報告書処理（Gemini PDF→Markdown変換）")
    api_key = os.environ.get("GOOGLE_API_KEY")
    reports = load_all_securities_reports(api_key)
    
    # 3. 企業マスタの作成
    print("\n[Step 1.3] 企業マスタ作成")
    masters = create_company_master(df, reports)
    
    # 4. データの保存
    print("\n[Step 1.4] データ保存")
    save_processed_data(df, summary, masters, reports)
    
    # 5. データ概要の表示
    print_data_overview(df, masters)
    
    print("Step 1 完了！")
    return df, summary, masters, reports


if __name__ == "__main__":
    main()

