"""
Step 7: 効果試算・ロードマップ
==============================
- 提案施策の定量効果試算
- KPI設定
- 実行ロードマップの詳細化
- リスクと対策
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import google.generativeai as genai


# =============================================================================
# 設定
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = BASE_DIR / "analysis"

# 入力ディレクトリ
STRATEGY_DIR = ANALYSIS_DIR / "strategy"

# 出力ディレクトリ
ROADMAP_DIR = ANALYSIS_DIR / "roadmap"
ROADMAP_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_LOG_DIR = BASE_DIR / "output" / "prompt_logs"
PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_MODEL = "gemini-3-pro-preview"


# =============================================================================
# データクラス
# =============================================================================
@dataclass
class EffectEstimation:
    """効果試算結果"""
    company_code: str
    
    # 財務効果
    revenue_effect: dict = field(default_factory=dict)
    profit_effect: dict = field(default_factory=dict)
    cost_reduction: dict = field(default_factory=dict)
    
    # 投資計画
    investment_plan: list[dict] = field(default_factory=list)
    roi_analysis: dict = field(default_factory=dict)
    
    # KPI設定
    kpis: list[dict] = field(default_factory=list)
    
    # 詳細ロードマップ
    roadmap: list[dict] = field(default_factory=list)
    milestones: list[dict] = field(default_factory=list)
    
    # リスクと対策
    risks_and_mitigations: list[dict] = field(default_factory=list)
    
    # サマリー
    effect_summary: str = ""


@dataclass
class PromptLogEntry:
    timestamp: str
    company_code: str
    purpose: str
    input_prompt: str
    output_response: str
    model: str = GEMINI_MODEL


class PromptLogger:
    def __init__(self, log_name: str = "step7"):
        self.logs: list[PromptLogEntry] = []
        self.log_file = PROMPT_LOG_DIR / f"prompt_log_{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def add_log(self, entry: PromptLogEntry):
        self.logs.append(entry)
        logs_dict = [asdict(log) for log in self.logs]
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs_dict, f, ensure_ascii=False, indent=2)
    
    def export_to_text(self) -> str:
        text = "=" * 80 + "\n"
        text += "Step 7: 効果試算 プロンプトログ\n"
        text += f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += "=" * 80 + "\n\n"
        
        for i, log in enumerate(self.logs, 1):
            text += f"--- ログ #{i} ---\n"
            text += f"日時: {log.timestamp}\n"
            text += f"対象企業: {log.company_code}\n"
            text += f"目的: {log.purpose}\n"
            text += f"\n【入力プロンプト】\n{log.input_prompt}\n"
            text += f"\n【出力結果】\n{log.output_response}\n"
            text += "\n" + "-" * 80 + "\n\n"
        
        return text


# =============================================================================
# データ読み込み
# =============================================================================
def load_strategy_results() -> dict:
    """戦略策定結果を読み込み"""
    json_path = STRATEGY_DIR / "strategy_results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_financial_analysis() -> dict:
    """財務分析結果を読み込み"""
    json_path = ANALYSIS_DIR / "reports" / "analysis_summary_for_ai.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_company_master() -> dict:
    """企業マスタを読み込み"""
    json_path = PROCESSED_DIR / "company_master.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            masters = json.load(f)
            return {m["code"]: m for m in masters}
    return {}


# =============================================================================
# 効果試算クラス
# =============================================================================
class EffectEstimator:
    """効果試算クラス"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY が設定されていません。")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.prompt_logger = PromptLogger("step7")
        
        self.strategy_data = load_strategy_results()
        self.financial_data = load_financial_analysis()
        self.company_master = load_company_master()
        
        # 有報分析結果を読み込み（Step 3の出力）
        self.securities_data = {}
        securities_path = ANALYSIS_DIR / "llm_analysis" / "securities_report_analysis.json"
        if securities_path.exists():
            with open(securities_path, "r", encoding="utf-8") as f:
                self.securities_data = json.load(f)
    
    def _call_gemini(self, prompt: str, company_code: str, purpose: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            result = response.text
            
            self.prompt_logger.add_log(PromptLogEntry(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                company_code=company_code,
                purpose=purpose,
                input_prompt=prompt,
                output_response=result,
            ))
            
            return result
        except Exception as e:
            print(f"[ERROR] API呼び出しエラー: {e}")
            time.sleep(2)
            return ""
    
    def _parse_json_response(self, response: str) -> dict:
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {}
    
    def estimate_effects(self, company_code: str) -> EffectEstimation:
        """効果試算を実行"""
        print(f"[INFO] 効果試算開始: {company_code}")
        
        strategy = self.strategy_data.get(company_code, {})
        financial = self.financial_data.get(company_code, {})
        master = self.company_master.get(company_code, {})
        securities = self.securities_data.get(company_code, {})
        
        # 現在の財務状況
        latest_revenue = financial.get('業績推移', {}).get('売上高_億円', [0])[-1] if financial else 0
        latest_profit = financial.get('業績推移', {}).get('営業利益_億円', [0])[-1] if financial else 0
        
        context = f"""
【企業情報】
- コード: {company_code}
- 所在地: {master.get('location', '')}
- 業種: {master.get('industry', '')}
- 従業員数: {master.get('employees', '')}名
- 現在の売上高: {latest_revenue:.1f}億円
- 現在の営業利益: {latest_profit:.1f}億円

【策定済み戦略】
- 戦略方向性: {strategy.get('strategic_direction', '')}
- 成長戦略: {strategy.get('growth_strategies', [])}
- DX施策: {strategy.get('dx_measures', [])}
- GX施策: {strategy.get('gx_measures', [])}
- 人材施策: {strategy.get('hr_measures', [])}

【中期経営計画の目標数値（効果試算の基準）】
{securities.get('medium_term_plan_targets', '情報なし')}

【企業自身の設備投資計画（既存投資との整合用）】
{securities.get('capex_plan', '情報なし')}
"""
        
        # 1. 財務効果試算
        effect_prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下の企業情報と戦略に基づき、定量的な効果を試算してください。

{context}

【試算の前提】
- 建設業界の一般的なベンチマークを参考にしてください
- 保守的なシナリオで試算してください
- 試算根拠を明示してください

【出力形式】
```json
{{
    "revenue_effect": {{
        "year1": {{"amount": "金額（億円）", "growth_rate": "成長率（%）", "basis": "試算根拠"}},
        "year3": {{"amount": "金額（億円）", "growth_rate": "成長率（%）", "basis": "試算根拠"}},
        "year5": {{"amount": "金額（億円）", "growth_rate": "成長率（%）", "basis": "試算根拠"}}
    }},
    "profit_effect": {{
        "year1": {{"amount": "金額（億円）", "margin": "利益率（%）", "basis": "試算根拠"}},
        "year3": {{"amount": "金額（億円）", "margin": "利益率（%）", "basis": "試算根拠"}},
        "year5": {{"amount": "金額（億円）", "margin": "利益率（%）", "basis": "試算根拠"}}
    }},
    "cost_reduction": {{
        "dx_effect": "DXによるコスト削減額（億円/年）",
        "productivity_improvement": "生産性向上効果（%）",
        "basis": "試算根拠"
    }}
}}
```
"""
        
        effect_response = self._call_gemini(effect_prompt, company_code, "財務効果試算")
        effect_data = self._parse_json_response(effect_response)
        
        time.sleep(1)
        
        # 2. 投資計画・ROI
        investment_prompt = f"""以下の戦略を実行するための投資計画とROI分析を行ってください。

{context}

【出力形式】
```json
{{
    "investment_plan": [
        {{
            "item": "投資項目",
            "category": "DX/GX/人材/設備",
            "amount": "投資額（百万円）",
            "timing": "投資時期",
            "payback_period": "回収期間"
        }}
    ],
    "roi_analysis": {{
        "total_investment": "総投資額（億円）",
        "expected_return": "期待リターン（億円/年）",
        "roi_rate": "ROI（%）",
        "payback_years": "投資回収期間（年）"
    }}
}}
```
"""
        
        investment_response = self._call_gemini(investment_prompt, company_code, "投資計画策定")
        investment_data = self._parse_json_response(investment_response)
        
        time.sleep(1)
        
        # 3. KPI設定
        kpi_prompt = f"""以下の戦略を管理するためのKPIを設定してください。

【戦略概要】
{context}

【KPI設定の観点】
- 財務KPI（売上、利益、ROE等）
- 事業KPI（受注、完工、生産性等）
- DX/GX KPI（デジタル化率、CO2削減等）
- 人材KPI（採用、定着、資格取得等）

【出力形式】
```json
{{
    "kpis": [
        {{
            "name": "KPI名",
            "category": "財務/事業/DX/GX/人材",
            "current": "現在値",
            "target_year1": "1年目目標",
            "target_year3": "3年目目標",
            "target_year5": "5年目目標",
            "measurement": "測定方法"
        }}
    ]
}}
```
"""
        
        kpi_response = self._call_gemini(kpi_prompt, company_code, "KPI設定")
        kpi_data = self._parse_json_response(kpi_response)
        
        time.sleep(1)
        
        # 4. 詳細ロードマップ
        roadmap_prompt = f"""以下の戦略の詳細な実行ロードマップを作成してください。

【戦略概要】
- 短期アクション: {strategy.get('short_term_actions', [])}
- 中期アクション: {strategy.get('medium_term_actions', [])}
- 長期アクション: {strategy.get('long_term_actions', [])}

【出力形式】
```json
{{
    "roadmap": [
        {{
            "phase": "Phase名",
            "period": "期間（例：2025年4月〜2025年9月）",
            "theme": "フェーズテーマ",
            "actions": ["アクション1", "アクション2"],
            "deliverables": ["成果物1", "成果物2"]
        }}
    ],
    "milestones": [
        {{
            "name": "マイルストーン名",
            "target_date": "達成目標時期",
            "criteria": "達成基準",
            "importance": "高/中/低"
        }}
    ]
}}
```
"""
        
        roadmap_response = self._call_gemini(roadmap_prompt, company_code, "ロードマップ策定")
        roadmap_data = self._parse_json_response(roadmap_response)
        
        time.sleep(1)
        
        # 5. リスクと対策
        risk_prompt = f"""以下の戦略を実行する上でのリスクと対策を洗い出してください。

【戦略概要】
{context}

【リスク分類】
- 実行リスク（リソース不足、スキル不足等）
- 市場リスク（需要変動、競合等）
- 財務リスク（投資回収、資金繰り等）
- 外部リスク（規制、経済環境等）

【出力形式】
```json
{{
    "risks_and_mitigations": [
        {{
            "risk": "リスク内容",
            "category": "実行/市場/財務/外部",
            "probability": "高/中/低",
            "impact": "高/中/低",
            "mitigation": "対策",
            "contingency": "発生時の対応"
        }}
    ]
}}
```
"""
        
        risk_response = self._call_gemini(risk_prompt, company_code, "リスク分析")
        risk_data = self._parse_json_response(risk_response)
        
        time.sleep(1)
        
        # 6. 効果サマリー
        summary_prompt = f"""以下の効果試算結果を、300字程度のサマリーにまとめてください。
投資対効果と実現に向けたポイントがわかるように簡潔にまとめてください。

【財務効果】
{json.dumps(effect_data, ensure_ascii=False)}

【投資計画】
{json.dumps(investment_data, ensure_ascii=False)}

【主要KPI】
{json.dumps(kpi_data.get('kpis', [])[:5], ensure_ascii=False)}

サマリーテキストのみを出力してください。
"""
        
        summary = self._call_gemini(summary_prompt, company_code, "効果サマリー")
        
        # 結果構築
        result = EffectEstimation(
            company_code=company_code,
            revenue_effect=effect_data.get("revenue_effect", {}),
            profit_effect=effect_data.get("profit_effect", {}),
            cost_reduction=effect_data.get("cost_reduction", {}),
            investment_plan=investment_data.get("investment_plan", []),
            roi_analysis=investment_data.get("roi_analysis", {}),
            kpis=kpi_data.get("kpis", []),
            roadmap=roadmap_data.get("roadmap", []),
            milestones=roadmap_data.get("milestones", []),
            risks_and_mitigations=risk_data.get("risks_and_mitigations", []),
            effect_summary=summary.strip()
        )
        
        print(f"[INFO] 効果試算完了: {company_code}")
        return result
    
    def estimate_all_companies(self, company_codes: list[str]) -> dict[str, EffectEstimation]:
        """全企業の効果試算を実行"""
        results = {}
        
        for i, code in enumerate(company_codes, 1):
            print(f"\n[{i}/{len(company_codes)}] {code}")
            try:
                result = self.estimate_effects(code)
                results[code] = result
            except Exception as e:
                print(f"[ERROR] {code}: {e}")
                results[code] = EffectEstimation(company_code=code)
            
            if i < len(company_codes):
                time.sleep(2)
        
        return results


# =============================================================================
# 結果出力
# =============================================================================
def save_estimation_results(results: dict[str, EffectEstimation]):
    """効果試算結果を保存"""
    
    # JSON形式
    all_results = {code: asdict(result) for code, result in results.items()}
    json_path = ROADMAP_DIR / "effect_estimation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {json_path}")
    
    # 企業別テキストレポート
    for code, result in results.items():
        report = generate_estimation_report(result)
        report_path = ROADMAP_DIR / f"{code}_roadmap.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
    print(f"[SAVE] 企業別ロードマップ: {len(results)}社")


def generate_estimation_report(result: EffectEstimation) -> str:
    """効果試算レポートを生成"""
    report = f"""
================================================================================
効果試算・ロードマップ: {result.company_code}
================================================================================

【効果サマリー】
{result.effect_summary}

================================================================================
【財務効果予測】
================================================================================

■ 売上高効果
"""
    
    for year, data in result.revenue_effect.items():
        if isinstance(data, dict):
            report += f"- {year}: {data.get('amount', '-')} ({data.get('growth_rate', '-')})\n"
            report += f"  根拠: {data.get('basis', '-')}\n"
    
    report += """
■ 利益効果
"""
    for year, data in result.profit_effect.items():
        if isinstance(data, dict):
            report += f"- {year}: {data.get('amount', '-')} (利益率 {data.get('margin', '-')})\n"
            report += f"  根拠: {data.get('basis', '-')}\n"
    
    report += f"""
■ コスト削減効果
- DX効果: {result.cost_reduction.get('dx_effect', '-')}
- 生産性向上: {result.cost_reduction.get('productivity_improvement', '-')}
- 根拠: {result.cost_reduction.get('basis', '-')}

================================================================================
【投資計画】
================================================================================
"""
    
    for i, inv in enumerate(result.investment_plan, 1):
        if isinstance(inv, dict):
            report += f"""
{i}. {inv.get('item', '投資項目')} [{inv.get('category', '')}]
   投資額: {inv.get('amount', '-')}百万円
   時期: {inv.get('timing', '-')}
   回収期間: {inv.get('payback_period', '-')}
"""
    
    roi = result.roi_analysis
    if roi:
        report += f"""
■ ROI分析
- 総投資額: {roi.get('total_investment', '-')}億円
- 期待リターン: {roi.get('expected_return', '-')}億円/年
- ROI: {roi.get('roi_rate', '-')}%
- 投資回収期間: {roi.get('payback_years', '-')}年

================================================================================
【KPI設定】
================================================================================
"""
    
    for kpi in result.kpis:
        if isinstance(kpi, dict):
            report += f"""
■ {kpi.get('name', 'KPI')} [{kpi.get('category', '')}]
  現在値: {kpi.get('current', '-')}
  1年目: {kpi.get('target_year1', '-')}
  3年目: {kpi.get('target_year3', '-')}
  5年目: {kpi.get('target_year5', '-')}
  測定方法: {kpi.get('measurement', '-')}
"""
    
    report += """
================================================================================
【実行ロードマップ】
================================================================================
"""
    
    for phase in result.roadmap:
        if isinstance(phase, dict):
            report += f"""
【{phase.get('phase', 'Phase')}】{phase.get('period', '')}
テーマ: {phase.get('theme', '')}
アクション:
"""
            for action in phase.get('actions', []):
                report += f"  - {action}\n"
            report += "成果物:\n"
            for deliverable in phase.get('deliverables', []):
                report += f"  - {deliverable}\n"
    
    report += """
■ マイルストーン
"""
    for ms in result.milestones:
        if isinstance(ms, dict):
            report += f"- {ms.get('target_date', '')}: {ms.get('name', '')} [{ms.get('importance', '')}]\n"
            report += f"  達成基準: {ms.get('criteria', '')}\n"
    
    report += """
================================================================================
【リスクと対策】
================================================================================
"""
    
    for risk in result.risks_and_mitigations:
        if isinstance(risk, dict):
            report += f"""
■ {risk.get('risk', 'リスク')} [{risk.get('category', '')}]
  発生確率: {risk.get('probability', '-')} / 影響度: {risk.get('impact', '-')}
  対策: {risk.get('mitigation', '-')}
  発生時対応: {risk.get('contingency', '-')}
"""
    
    return report


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print("=" * 60)
    print("Step 7: 効果試算・ロードマップ")
    print("=" * 60 + "\n")
    
    # 企業コードの取得
    company_master = load_company_master()
    company_codes = sorted(company_master.keys())
    print(f"[INFO] 対象企業: {len(company_codes)}社")
    
    # 戦略結果の確認
    strategy_data = load_strategy_results()
    if not strategy_data:
        print("[WARNING] 戦略策定結果がありません。Step 6を先に実行してください。")
    
    # API キー確認
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n[ERROR] GOOGLE_API_KEY が設定されていません。")
        return None
    
    # 効果試算
    estimator = EffectEstimator(api_key)
    results = estimator.estimate_all_companies(company_codes)
    
    # 結果保存
    save_estimation_results(results)
    
    # プロンプトログ保存
    log_text = estimator.prompt_logger.export_to_text()
    log_path = PROMPT_LOG_DIR / "step7_prompt_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)
    print(f"[SAVE] {log_path}")
    
    print("\n" + "=" * 60)
    print("Step 7 完了！")
    print(f"出力先: {ROADMAP_DIR}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

