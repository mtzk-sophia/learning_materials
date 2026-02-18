"""
Step 2: 財務分析
================
- 収益性分析（3年推移）
- 安全性分析
- 効率性分析
- キャッシュフロー分析
- 建設業特有の指標分析
- 企業間比較
- 分析レポート生成
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIバックエンドを使用しない
import japanize_matplotlib  # 日本語フォント対応


# =============================================================================
# 設定
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = BASE_DIR / "analysis"

# 出力ディレクトリの作成
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
(ANALYSIS_DIR / "charts").mkdir(exist_ok=True)
(ANALYSIS_DIR / "reports").mkdir(exist_ok=True)


# =============================================================================
# データ読み込み
# =============================================================================
def load_processed_data() -> tuple[pd.DataFrame, list[dict]]:
    """Step 1で生成した前処理済みデータを読み込む"""
    df = pd.read_csv(PROCESSED_DIR / "financial_data_processed.csv")
    df["コード"] = df["コード"].astype(str)
    
    with open(PROCESSED_DIR / "company_master.json", "r", encoding="utf-8") as f:
        masters = json.load(f)
    
    print(f"[INFO] データ読み込み完了: {len(df)}行, {len(masters)}社")
    return df, masters


# =============================================================================
# 財務分析クラス
# =============================================================================
@dataclass
class FinancialAnalysisResult:
    """財務分析結果"""
    company_code: str
    company_name: str  # 所在地 + 業種
    location: str
    industry: str
    
    # 基本情報
    employees: int
    capital: float
    market: str
    
    # 3年間の推移データ
    years: list[int] = field(default_factory=list)
    
    # 収益性指標（3年分）
    revenue: list[float] = field(default_factory=list)
    operating_income: list[float] = field(default_factory=list)
    net_income: list[float] = field(default_factory=list)
    operating_margin: list[float] = field(default_factory=list)
    net_margin: list[float] = field(default_factory=list)
    roe: list[float] = field(default_factory=list)
    roa: list[float] = field(default_factory=list)
    
    # 安全性指標（3年分）
    equity_ratio: list[float] = field(default_factory=list)
    current_ratio: list[float] = field(default_factory=list)
    debt_ratio: list[float] = field(default_factory=list)
    
    # 効率性指標（3年分）
    asset_turnover: list[float] = field(default_factory=list)
    revenue_per_employee: list[float] = field(default_factory=list)
    
    # キャッシュフロー（3年分）
    operating_cf: list[float] = field(default_factory=list)
    investing_cf: list[float] = field(default_factory=list)
    financing_cf: list[float] = field(default_factory=list)
    free_cf: list[float] = field(default_factory=list)
    cash_balance: list[float] = field(default_factory=list)
    
    # 建設業特有指標
    construction_profit_margin: list[float] = field(default_factory=list)
    
    # 成長率
    revenue_growth: list[Optional[float]] = field(default_factory=list)
    operating_income_growth: list[Optional[float]] = field(default_factory=list)
    
    # 分析コメント
    profitability_comment: str = ""
    safety_comment: str = ""
    efficiency_comment: str = ""
    cashflow_comment: str = ""
    overall_comment: str = ""
    
    # 強み・課題
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


class FinancialAnalyzer:
    """財務分析を実行するクラス"""
    
    def __init__(self, df: pd.DataFrame, masters: list[dict]):
        self.df = df
        self.masters = {m["code"]: m for m in masters}
        self.results: dict[str, FinancialAnalysisResult] = {}
        
        # 業界平均値を計算
        self._calculate_industry_averages()
    
    def _calculate_industry_averages(self):
        """業界平均値を計算"""
        latest_year = self.df["YEAR"].max()
        latest_df = self.df[self.df["YEAR"] == latest_year]
        
        self.industry_avg = {
            "売上高営業利益率": latest_df["売上高営業利益率"].mean(),
            "売上高当期純利益率": latest_df["売上高当期純利益率"].mean(),
            "ROE": latest_df["ROE"].mean(),
            "ROA": latest_df["ROA"].mean(),
            "自己資本比率": latest_df["自己資本比率"].mean(),
            "流動比率": latest_df["流動比率"].mean(),
            "総資産回転率": latest_df["総資産回転率"].mean(),
        }
        
        print("[INFO] 業界平均値を計算:")
        for key, val in self.industry_avg.items():
            print(f"  - {key}: {val:.2f}")
    
    def analyze_company(self, company_code: str) -> FinancialAnalysisResult:
        """1社分の財務分析を実行"""
        company_df = self.df[self.df["コード"] == company_code].sort_values("YEAR")
        master = self.masters[company_code]
        
        result = FinancialAnalysisResult(
            company_code=company_code,
            company_name=f"{master['location']}・{master['industry']}",
            location=master["location"],
            industry=master["industry"],
            employees=master["employees"],
            capital=master["capital"],
            market=master["market"],
        )
        
        # 年度リスト
        result.years = company_df["YEAR"].tolist()
        
        # 収益性指標
        result.revenue = (company_df["売上高"] / 1e8).tolist()
        result.operating_income = (company_df["営業利益"] / 1e8).tolist()
        result.net_income = (company_df["当期純利益"] / 1e8).tolist()
        result.operating_margin = company_df["売上高営業利益率"].tolist()
        result.net_margin = company_df["売上高当期純利益率"].tolist()
        result.roe = company_df["ROE"].tolist()
        result.roa = company_df["ROA"].tolist()
        
        # 安全性指標
        result.equity_ratio = company_df["自己資本比率"].tolist()
        result.current_ratio = company_df["流動比率"].tolist()
        result.debt_ratio = company_df["負債比率"].tolist()
        
        # 効率性指標
        result.asset_turnover = company_df["総資産回転率"].tolist()
        result.revenue_per_employee = company_df["従業員一人当たり売上高_百万円"].tolist()
        
        # キャッシュフロー
        result.operating_cf = (company_df["営業活動によるキャッシュ・フロー"] / 1e8).tolist()
        result.investing_cf = (company_df["投資活動によるキャッシュ・フロー"] / 1e8).tolist()
        result.financing_cf = (company_df["財務活動によるキャッシュ・フロー"] / 1e8).tolist()
        result.free_cf = (company_df["フリーキャッシュフロー"] / 1e8).tolist()
        result.cash_balance = (company_df["現金及び現金同等物期末残高"] / 1e8).tolist()
        
        # 建設業特有指標
        if "完成工事総利益率" in company_df.columns:
            result.construction_profit_margin = company_df["完成工事総利益率"].tolist()
        
        # 成長率
        result.revenue_growth = company_df["売上高_前年比"].tolist()
        result.operating_income_growth = company_df["営業利益_前年比"].tolist()
        
        # 分析コメントの生成
        self._generate_comments(result)
        
        # 強み・課題の抽出
        self._extract_strengths_weaknesses(result)
        
        self.results[company_code] = result
        return result
    
    def _generate_comments(self, result: FinancialAnalysisResult):
        """分析コメントを生成"""
        latest_idx = -1  # 最新年度
        
        # 収益性コメント
        latest_margin = result.operating_margin[latest_idx]
        avg_margin = self.industry_avg["売上高営業利益率"]
        margin_trend = "上昇" if len(result.operating_margin) > 1 and result.operating_margin[latest_idx] > result.operating_margin[0] else "下降"
        
        if latest_margin > avg_margin * 1.2:
            profitability_level = "業界平均を大きく上回る高収益"
        elif latest_margin > avg_margin:
            profitability_level = "業界平均を上回る収益性"
        elif latest_margin > avg_margin * 0.8:
            profitability_level = "業界平均並みの収益性"
        elif latest_margin > 0:
            profitability_level = "業界平均を下回る収益性"
        else:
            profitability_level = "営業赤字の状況"
        
        result.profitability_comment = f"""
【収益性分析】
- 営業利益率: {latest_margin:.1f}%（業界平均: {avg_margin:.1f}%）→ {profitability_level}
- ROE: {result.roe[latest_idx]:.1f}%、ROA: {result.roa[latest_idx]:.1f}%
- 3年間のトレンド: 営業利益率は{margin_trend}傾向
- 売上高: {result.revenue[0]:.1f}億円 → {result.revenue[latest_idx]:.1f}億円（3年間）
""".strip()
        
        # 安全性コメント
        latest_equity = result.equity_ratio[latest_idx]
        latest_current = result.current_ratio[latest_idx]
        
        if latest_equity > 50:
            equity_level = "非常に健全（自己資本比率50%超）"
        elif latest_equity > 30:
            equity_level = "健全な水準（自己資本比率30-50%）"
        elif latest_equity > 20:
            equity_level = "やや低い水準（自己資本比率20-30%）"
        else:
            equity_level = "要注意の水準（自己資本比率20%未満）"
        
        result.safety_comment = f"""
【安全性分析】
- 自己資本比率: {latest_equity:.1f}% → {equity_level}
- 流動比率: {latest_current:.1f}%（100%以上が望ましい）
- 負債比率: {result.debt_ratio[latest_idx]:.1f}%
""".strip()
        
        # 効率性コメント
        latest_turnover = result.asset_turnover[latest_idx]
        avg_turnover = self.industry_avg["総資産回転率"]
        
        result.efficiency_comment = f"""
【効率性分析】
- 総資産回転率: {latest_turnover:.2f}回（業界平均: {avg_turnover:.2f}回）
- 従業員一人当たり売上高: {result.revenue_per_employee[latest_idx]:.1f}百万円
- 従業員数: {result.employees}名
""".strip()
        
        # キャッシュフローコメント
        latest_ocf = result.operating_cf[latest_idx]
        latest_fcf = result.free_cf[latest_idx]
        
        if latest_ocf > 0 and latest_fcf > 0:
            cf_status = "営業CF・FCFともにプラスで健全"
        elif latest_ocf > 0 and latest_fcf < 0:
            cf_status = "営業CFプラスだが積極投資によりFCFマイナス"
        elif latest_ocf < 0 and latest_fcf < 0:
            cf_status = "営業CFマイナスで資金繰りに注意が必要"
        else:
            cf_status = "キャッシュフローの状況を要確認"
        
        result.cashflow_comment = f"""
【キャッシュフロー分析】
- 営業CF: {latest_ocf:.1f}億円
- 投資CF: {result.investing_cf[latest_idx]:.1f}億円
- 財務CF: {result.financing_cf[latest_idx]:.1f}億円
- フリーCF: {latest_fcf:.1f}億円 → {cf_status}
- 現金残高: {result.cash_balance[latest_idx]:.1f}億円
""".strip()
        
        # 総合コメント
        result.overall_comment = f"""
【総合評価】
{result.company_name}（コード: {result.company_code}）

■ 企業概要
- 所在地: {result.location}
- 業種: {result.industry}
- 市場: {result.market}
- 従業員数: {result.employees}名
- 資本金: {result.capital}億円

■ 業績推移（{result.years[0]}年度→{result.years[-1]}年度）
- 売上高: {result.revenue[0]:.1f}億円 → {result.revenue[-1]:.1f}億円
- 営業利益: {result.operating_income[0]:.1f}億円 → {result.operating_income[-1]:.1f}億円
- 当期純利益: {result.net_income[0]:.1f}億円 → {result.net_income[-1]:.1f}億円
""".strip()
    
    def _extract_strengths_weaknesses(self, result: FinancialAnalysisResult):
        """強み・課題を抽出"""
        latest_idx = -1
        
        # 強み
        strengths = []
        weaknesses = []
        
        # 収益性の評価
        if result.operating_margin[latest_idx] > self.industry_avg["売上高営業利益率"]:
            strengths.append(f"業界平均を上回る収益性（営業利益率{result.operating_margin[latest_idx]:.1f}%）")
        elif result.operating_margin[latest_idx] < 0:
            weaknesses.append(f"営業赤字の状態（営業利益率{result.operating_margin[latest_idx]:.1f}%）")
        elif result.operating_margin[latest_idx] < self.industry_avg["売上高営業利益率"] * 0.7:
            weaknesses.append(f"収益性が業界平均を下回る（営業利益率{result.operating_margin[latest_idx]:.1f}%）")
        
        # ROEの評価
        if result.roe[latest_idx] > 10:
            strengths.append(f"高いROE（{result.roe[latest_idx]:.1f}%）")
        elif result.roe[latest_idx] < 0:
            weaknesses.append(f"ROEがマイナス（{result.roe[latest_idx]:.1f}%）")
        
        # 安全性の評価
        if result.equity_ratio[latest_idx] > 50:
            strengths.append(f"高い自己資本比率（{result.equity_ratio[latest_idx]:.1f}%）による財務安定性")
        elif result.equity_ratio[latest_idx] < 20:
            weaknesses.append(f"自己資本比率が低い（{result.equity_ratio[latest_idx]:.1f}%）")
        
        # 流動性の評価
        if result.current_ratio[latest_idx] > 200:
            strengths.append(f"十分な流動性（流動比率{result.current_ratio[latest_idx]:.1f}%）")
        elif result.current_ratio[latest_idx] < 100:
            weaknesses.append(f"流動性に懸念（流動比率{result.current_ratio[latest_idx]:.1f}%）")
        
        # キャッシュフローの評価
        if result.operating_cf[latest_idx] > 0 and result.free_cf[latest_idx] > 0:
            strengths.append("安定したキャッシュ創出力")
        elif result.operating_cf[latest_idx] < 0:
            weaknesses.append("営業キャッシュフローがマイナス")
        
        # 成長性の評価
        if result.revenue_growth[latest_idx] is not None and result.revenue_growth[latest_idx] > 10:
            strengths.append(f"高い売上成長率（前年比+{result.revenue_growth[latest_idx]:.1f}%）")
        elif result.revenue_growth[latest_idx] is not None and result.revenue_growth[latest_idx] < -10:
            weaknesses.append(f"売上高の大幅減少（前年比{result.revenue_growth[latest_idx]:.1f}%）")
        
        # 効率性の評価
        if result.asset_turnover[latest_idx] > self.industry_avg["総資産回転率"] * 1.2:
            strengths.append(f"高い資産効率（総資産回転率{result.asset_turnover[latest_idx]:.2f}回）")
        
        result.strengths = strengths if strengths else ["特筆すべき強みを要検討"]
        result.weaknesses = weaknesses if weaknesses else ["大きな課題は見られない"]
    
    def analyze_all_companies(self) -> dict[str, FinancialAnalysisResult]:
        """全企業の分析を実行"""
        for code in self.masters.keys():
            print(f"[INFO] 分析中: {code}")
            self.analyze_company(code)
        
        print(f"[INFO] 全{len(self.results)}社の分析完了")
        return self.results
    
    def create_comparison_table(self) -> pd.DataFrame:
        """企業間比較表を作成"""
        data = []
        for code, result in self.results.items():
            data.append({
                "コード": code,
                "所在地": result.location,
                "業種": result.industry,
                "従業員数": result.employees,
                "売上高_億円": result.revenue[-1],
                "営業利益_億円": result.operating_income[-1],
                "当期純利益_億円": result.net_income[-1],
                "営業利益率_%": result.operating_margin[-1],
                "ROE_%": result.roe[-1],
                "ROA_%": result.roa[-1],
                "自己資本比率_%": result.equity_ratio[-1],
                "流動比率_%": result.current_ratio[-1],
                "総資産回転率": result.asset_turnover[-1],
                "フリーCF_億円": result.free_cf[-1],
                "売上高成長率_%": result.revenue_growth[-1],
            })
        
        comparison_df = pd.DataFrame(data)
        return comparison_df


# =============================================================================
# グラフ生成
# =============================================================================
class ChartGenerator:
    """グラフ生成クラス"""
    
    def __init__(self, results: dict[str, FinancialAnalysisResult]):
        self.results = results
        self.chart_dir = ANALYSIS_DIR / "charts"
    
    def generate_company_charts(self, company_code: str):
        """企業別のグラフを生成"""
        result = self.results[company_code]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{result.company_name}（{company_code}）財務分析", fontsize=14, fontweight='bold')
        
        years = result.years
        
        # 1. 売上・利益推移
        ax1 = axes[0, 0]
        x = range(len(years))
        width = 0.25
        ax1.bar([i - width for i in x], result.revenue, width, label='売上高', color='steelblue')
        ax1.bar([i for i in x], result.operating_income, width, label='営業利益', color='forestgreen')
        ax1.bar([i + width for i in x], result.net_income, width, label='当期純利益', color='coral')
        ax1.set_xlabel('年度')
        ax1.set_ylabel('金額（億円）')
        ax1.set_title('売上高・利益推移')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 収益性指標推移
        ax2 = axes[0, 1]
        ax2.plot(years, result.operating_margin, marker='o', label='営業利益率', color='forestgreen')
        ax2.plot(years, result.roe, marker='s', label='ROE', color='coral')
        ax2.plot(years, result.roa, marker='^', label='ROA', color='steelblue')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('年度')
        ax2.set_ylabel('比率（%）')
        ax2.set_title('収益性指標推移')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. 安全性指標
        ax3 = axes[1, 0]
        ax3.plot(years, result.equity_ratio, marker='o', label='自己資本比率', color='steelblue')
        ax3.plot(years, result.current_ratio, marker='s', label='流動比率（右軸）', color='forestgreen')
        ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='流動比率100%ライン')
        ax3.set_xlabel('年度')
        ax3.set_ylabel('比率（%）')
        ax3.set_title('安全性指標推移')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. キャッシュフロー
        ax4 = axes[1, 1]
        width = 0.2
        ax4.bar([i - width for i in x], result.operating_cf, width, label='営業CF', color='steelblue')
        ax4.bar([i for i in x], result.investing_cf, width, label='投資CF', color='coral')
        ax4.bar([i + width for i in x], result.financing_cf, width, label='財務CF', color='forestgreen')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.set_xlabel('年度')
        ax4.set_ylabel('金額（億円）')
        ax4.set_title('キャッシュフロー推移')
        ax4.set_xticks(x)
        ax4.set_xticklabels(years)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.chart_dir / f"{company_code}_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVE] {company_code}_analysis.png")
    
    def generate_comparison_chart(self):
        """企業間比較グラフを生成"""
        codes = list(self.results.keys())
        labels = [f"{self.results[c].location}\n{c}" for c in codes]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('企業間比較（最新年度）', fontsize=14, fontweight='bold')
        
        # 1. 売上高比較
        ax1 = axes[0, 0]
        revenues = [self.results[c].revenue[-1] for c in codes]
        colors = plt.cm.Set3(np.linspace(0, 1, len(codes)))
        ax1.barh(labels, revenues, color=colors)
        ax1.set_xlabel('売上高（億円）')
        ax1.set_title('売上高比較')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. 営業利益率比較
        ax2 = axes[0, 1]
        margins = [self.results[c].operating_margin[-1] for c in codes]
        colors_margin = ['forestgreen' if m > 0 else 'coral' for m in margins]
        ax2.barh(labels, margins, color=colors_margin)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_xlabel('営業利益率（%）')
        ax2.set_title('営業利益率比較')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. 自己資本比率比較
        ax3 = axes[1, 0]
        equity_ratios = [self.results[c].equity_ratio[-1] for c in codes]
        ax3.barh(labels, equity_ratios, color='steelblue')
        ax3.axvline(x=30, color='orange', linestyle='--', alpha=0.7, label='30%ライン')
        ax3.set_xlabel('自己資本比率（%）')
        ax3.set_title('自己資本比率比較')
        ax3.legend()
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. ROE比較
        ax4 = axes[1, 1]
        roes = [self.results[c].roe[-1] for c in codes]
        colors_roe = ['forestgreen' if r > 0 else 'coral' for r in roes]
        ax4.barh(labels, roes, color=colors_roe)
        ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax4.axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='10%ライン')
        ax4.set_xlabel('ROE（%）')
        ax4.set_title('ROE比較')
        ax4.legend()
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.chart_dir / "comparison_all.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("[SAVE] comparison_all.png")
    
    def generate_all_charts(self):
        """全てのグラフを生成"""
        for code in self.results.keys():
            self.generate_company_charts(code)
        
        self.generate_comparison_chart()


# =============================================================================
# レポート生成
# =============================================================================
class ReportGenerator:
    """分析レポート生成クラス"""
    
    def __init__(self, results: dict[str, FinancialAnalysisResult], comparison_df: pd.DataFrame):
        self.results = results
        self.comparison_df = comparison_df
        self.report_dir = ANALYSIS_DIR / "reports"
    
    def generate_company_report(self, company_code: str) -> str:
        """企業別レポートを生成"""
        result = self.results[company_code]
        
        report = f"""
================================================================================
財務分析レポート: {result.company_name}（コード: {company_code}）
================================================================================

{result.overall_comment}

--------------------------------------------------------------------------------
{result.profitability_comment}

--------------------------------------------------------------------------------
{result.safety_comment}

--------------------------------------------------------------------------------
{result.efficiency_comment}

--------------------------------------------------------------------------------
{result.cashflow_comment}

--------------------------------------------------------------------------------
【強み】
{chr(10).join('- ' + s for s in result.strengths)}

【課題・リスク】
{chr(10).join('- ' + w for w in result.weaknesses)}

================================================================================
"""
        return report
    
    def generate_all_reports(self):
        """全企業のレポートを生成"""
        all_reports = ""
        
        for code, result in self.results.items():
            report = self.generate_company_report(code)
            all_reports += report + "\n"
            
            # 個別ファイル保存
            with open(self.report_dir / f"{code}_analysis.txt", "w", encoding="utf-8") as f:
                f.write(report)
            print(f"[SAVE] {code}_analysis.txt")
        
        # 全社まとめファイル
        with open(self.report_dir / "all_companies_analysis.txt", "w", encoding="utf-8") as f:
            f.write(all_reports)
        print("[SAVE] all_companies_analysis.txt")
        
        # 比較表の保存
        self.comparison_df.to_csv(
            self.report_dir / "company_comparison.csv",
            index=False,
            encoding="utf-8-sig"
        )
        print("[SAVE] company_comparison.csv")
    
    def generate_summary_for_ai(self) -> dict:
        """生成AI用のサマリーデータを作成"""
        summary = {}
        
        for code, result in self.results.items():
            summary[code] = {
                "基本情報": {
                    "所在地": result.location,
                    "業種": result.industry,
                    "従業員数": result.employees,
                    "資本金_億円": result.capital,
                    "市場": result.market,
                },
                "業績推移": {
                    "年度": result.years,
                    "売上高_億円": result.revenue,
                    "営業利益_億円": result.operating_income,
                    "当期純利益_億円": result.net_income,
                },
                "収益性指標": {
                    "営業利益率_%": result.operating_margin,
                    "ROE_%": result.roe,
                    "ROA_%": result.roa,
                },
                "安全性指標": {
                    "自己資本比率_%": result.equity_ratio,
                    "流動比率_%": result.current_ratio,
                },
                "キャッシュフロー_億円": {
                    "営業CF": result.operating_cf,
                    "投資CF": result.investing_cf,
                    "財務CF": result.financing_cf,
                    "フリーCF": result.free_cf,
                },
                "分析結果": {
                    "強み": result.strengths,
                    "課題": result.weaknesses,
                },
            }
        
        # JSON保存
        with open(self.report_dir / "analysis_summary_for_ai.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[SAVE] analysis_summary_for_ai.json")
        
        return summary


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print("="*60)
    print("Step 2: 財務分析")
    print("="*60 + "\n")
    
    # 1. データ読み込み
    print("[Step 2.1] データ読み込み")
    df, masters = load_processed_data()
    
    # 2. 財務分析実行
    print("\n[Step 2.2] 財務分析実行")
    analyzer = FinancialAnalyzer(df, masters)
    results = analyzer.analyze_all_companies()
    comparison_df = analyzer.create_comparison_table()
    
    # 3. グラフ生成
    print("\n[Step 2.3] グラフ生成")
    chart_gen = ChartGenerator(results)
    chart_gen.generate_all_charts()
    
    # 4. レポート生成
    print("\n[Step 2.4] レポート生成")
    report_gen = ReportGenerator(results, comparison_df)
    report_gen.generate_all_reports()
    ai_summary = report_gen.generate_summary_for_ai()
    
    # 5. 結果サマリー表示
    print("\n" + "="*60)
    print("分析結果サマリー")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("Step 2 完了！")
    print(f"出力先: {ANALYSIS_DIR}")
    print("="*60)
    
    return results, comparison_df, ai_summary


if __name__ == "__main__":
    main()

