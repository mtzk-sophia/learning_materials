"""
Step 6: 戦略策定・提案
======================
- SWOT分析結果を基にした成長戦略の策定
- GX/DX対応策
- 人材確保・定着策
- 具体的な施策の立案
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
SWOT_DIR = ANALYSIS_DIR / "swot"

# 出力ディレクトリ
STRATEGY_DIR = ANALYSIS_DIR / "strategy"
STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_LOG_DIR = BASE_DIR / "output" / "prompt_logs"
PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_MODEL = "gemini-3-pro-preview"


# =============================================================================
# データクラス
# =============================================================================
@dataclass
class StrategicProposal:
    """戦略提案"""
    company_code: str
    
    # 戦略の方向性
    strategic_direction: str = ""
    vision: str = ""
    
    # 成長戦略
    growth_strategies: list[dict] = field(default_factory=list)
    
    # DX施策
    dx_measures: list[dict] = field(default_factory=list)
    
    # GX施策
    gx_measures: list[dict] = field(default_factory=list)
    
    # 人材施策
    hr_measures: list[dict] = field(default_factory=list)
    
    # 財務改善施策
    financial_measures: list[dict] = field(default_factory=list)
    
    # 短期・中期・長期施策
    short_term_actions: list[dict] = field(default_factory=list)  # 1年以内
    medium_term_actions: list[dict] = field(default_factory=list)  # 1-3年
    long_term_actions: list[dict] = field(default_factory=list)  # 3-5年
    
    # 提案サマリー
    executive_summary: str = ""


@dataclass
class PromptLogEntry:
    timestamp: str
    company_code: str
    purpose: str
    input_prompt: str
    output_response: str
    model: str = GEMINI_MODEL


class PromptLogger:
    def __init__(self, log_name: str = "step6"):
        self.logs: list[PromptLogEntry] = []
        self.log_file = PROMPT_LOG_DIR / f"prompt_log_{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def add_log(self, entry: PromptLogEntry):
        self.logs.append(entry)
        logs_dict = [asdict(log) for log in self.logs]
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs_dict, f, ensure_ascii=False, indent=2)
    
    def export_to_text(self) -> str:
        text = "=" * 80 + "\n"
        text += "Step 6: 戦略策定 プロンプトログ\n"
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
def load_swot_results() -> dict:
    """SWOT分析結果を読み込み"""
    json_path = SWOT_DIR / "swot_analysis_results.json"
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
# 戦略策定クラス
# =============================================================================
class StrategyPlanner:
    """戦略策定クラス"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY が設定されていません。")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.prompt_logger = PromptLogger("step6")
        
        self.swot_data = load_swot_results()
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
    
    def plan_strategy(self, company_code: str) -> StrategicProposal:
        """戦略を策定"""
        print(f"[INFO] 戦略策定開始: {company_code}")
        
        swot = self.swot_data.get(company_code, {})
        financial = self.financial_data.get(company_code, {})
        master = self.company_master.get(company_code, {})
        securities = self.securities_data.get(company_code, {})
        
        context = f"""
【企業情報】
- コード: {company_code}
- 所在地: {master.get('location', '')}
- 業種: {master.get('industry', '')}
- 従業員数: {master.get('employees', '')}名

【SWOT分析結果】
- 強み: {swot.get('strengths', [])}
- 弱み: {swot.get('weaknesses', [])}
- 機会: {swot.get('opportunities', [])}
- 脅威: {swot.get('threats', [])}
- SO戦略: {swot.get('so_strategies', [])}
- WO戦略: {swot.get('wo_strategies', [])}
- ST戦略: {swot.get('st_strategies', [])}
- WT戦略: {swot.get('wt_strategies', [])}
- 優先課題: {swot.get('priority_issues', [])}

【財務状況】
- 業績推移: {financial.get('業績推移', {})}
- 収益性: {financial.get('収益性指標', {})}

【事業実績・計画（有報分析より）】
- 中期経営計画の目標数値: {securities.get('medium_term_plan_targets', '情報なし')}
- セグメント別業績: {securities.get('segment_performance', '情報なし')}
- 受注・販売実績: {securities.get('order_sales_record', '情報なし')}

【経営者の分析（インサイト）】
{securities.get('management_analysis', '情報なし')}

【設備投資の戦略分析】
{securities.get('capex_plan', '情報なし')}
"""
        
        # 1. 成長戦略の策定
        growth_prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下の企業情報とSWOT分析結果をもとに、成長戦略を策定してください。

{context}

【戦略策定の観点】
1. 売上拡大策（新規顧客開拓、事業領域拡大）
2. 収益性向上策（高付加価値化、コスト削減）
3. 事業ポートフォリオの最適化
4. 地域戦略（商圏拡大、地域深耕）

【出力形式】
JSON形式で出力してください。
```json
{{
    "strategic_direction": "全体の戦略方向性（100字程度）",
    "vision": "3-5年後の目指す姿（100字程度）",
    "growth_strategies": [
        {{
            "name": "戦略名",
            "description": "戦略の概要",
            "expected_effect": "期待効果",
            "priority": "高/中/低"
        }}
    ]
}}
```
"""
        
        growth_response = self._call_gemini(growth_prompt, company_code, "成長戦略策定")
        growth_data = self._parse_json_response(growth_response)
        
        time.sleep(1)
        
        # 2. DX施策
        dx_prompt = f"""あなたは建設DXの専門家です。
以下の建設会社に対し、具体的なDX施策を提案してください。

{context}

【DX施策の観点】
1. 施工DX（ICT施工、BIM/CIM、ドローン活用）
2. 業務効率化（基幹システム、ペーパーレス化）
3. 遠隔・省力化（遠隔臨場、自動化）
4. データ活用（原価管理、予測分析）

【出力形式】
```json
{{
    "dx_measures": [
        {{
            "name": "施策名",
            "category": "施工DX/業務効率化/遠隔・省力化/データ活用",
            "description": "施策の概要",
            "investment": "投資規模（大/中/小）",
            "timeline": "導入期間",
            "expected_effect": "期待効果"
        }}
    ]
}}
```
"""
        
        dx_response = self._call_gemini(dx_prompt, company_code, "DX施策策定")
        dx_data = self._parse_json_response(dx_response)
        
        time.sleep(1)
        
        # 3. GX施策
        gx_prompt = f"""あなたは建設GXの専門家です。
以下の建設会社に対し、具体的なGX施策を提案してください。

{context}

【GX施策の観点】
1. 施工時のCO2削減（低炭素資材、省エネ建機）
2. 環境配慮型建築（ZEB/ZEH、省エネ設計）
3. 再生可能エネルギー（太陽光、蓄電）
4. 循環型経済（廃棄物削減、リサイクル）

【出力形式】
```json
{{
    "gx_measures": [
        {{
            "name": "施策名",
            "category": "CO2削減/環境建築/再エネ/循環経済",
            "description": "施策の概要",
            "investment": "投資規模（大/中/小）",
            "co2_reduction": "CO2削減効果の見込み",
            "expected_effect": "その他期待効果"
        }}
    ]
}}
```
"""
        
        gx_response = self._call_gemini(gx_prompt, company_code, "GX施策策定")
        gx_data = self._parse_json_response(gx_response)
        
        time.sleep(1)
        
        # 4. 人材施策
        hr_prompt = f"""あなたは建設業の人事コンサルタントです。
以下の建設会社に対し、人材確保・定着策を提案してください。

{context}

【人材施策の観点】
1. 採用強化（新卒、中途、外国人材）
2. 定着・エンゲージメント向上
3. 技術承継・人材育成
4. 働き方改革（2024年問題対応）
5. ダイバーシティ推進

【出力形式】
```json
{{
    "hr_measures": [
        {{
            "name": "施策名",
            "category": "採用/定着/育成/働き方/ダイバーシティ",
            "description": "施策の概要",
            "target": "対象者",
            "timeline": "実施期間",
            "expected_effect": "期待効果"
        }}
    ]
}}
```
"""
        
        hr_response = self._call_gemini(hr_prompt, company_code, "人材施策策定")
        hr_data = self._parse_json_response(hr_response)
        
        time.sleep(1)
        
        # 5. アクションプラン（短期・中期・長期）
        action_prompt = f"""あなたは経営コンサルタントです。
これまでに策定した施策を、実行時期別に整理してください。

【策定済み施策】
- 成長戦略: {growth_data.get('growth_strategies', [])}
- DX施策: {dx_data.get('dx_measures', [])}
- GX施策: {gx_data.get('gx_measures', [])}
- 人材施策: {hr_data.get('hr_measures', [])}

【時間軸】
- 短期（1年以内）: すぐに着手可能、クイックウィン
- 中期（1-3年）: 体制構築、本格投資
- 長期（3-5年）: 成果刈り取り、次の成長

【出力形式】
```json
{{
    "short_term_actions": [
        {{"action": "アクション", "category": "成長/DX/GX/人材/財務", "kpi": "成果指標"}}
    ],
    "medium_term_actions": [
        {{"action": "アクション", "category": "成長/DX/GX/人材/財務", "kpi": "成果指標"}}
    ],
    "long_term_actions": [
        {{"action": "アクション", "category": "成長/DX/GX/人材/財務", "kpi": "成果指標"}}
    ]
}}
```
"""
        
        action_response = self._call_gemini(action_prompt, company_code, "アクションプラン策定")
        action_data = self._parse_json_response(action_response)
        
        time.sleep(1)
        
        # 6. エグゼクティブサマリー
        summary_prompt = f"""以下の戦略提案を、経営者向けのエグゼクティブサマリー（500字程度）にまとめてください。

【戦略の方向性】
{growth_data.get('strategic_direction', '')}

【目指す姿】
{growth_data.get('vision', '')}

【主要施策】
- 成長戦略: {len(growth_data.get('growth_strategies', []))}件
- DX施策: {len(dx_data.get('dx_measures', []))}件
- GX施策: {len(gx_data.get('gx_measures', []))}件
- 人材施策: {len(hr_data.get('hr_measures', []))}件

【優先アクション】
- 短期: {action_data.get('short_term_actions', [])[:3]}
- 中期: {action_data.get('medium_term_actions', [])[:2]}

簡潔で説得力のあるサマリーを作成してください。
"""
        
        summary = self._call_gemini(summary_prompt, company_code, "エグゼクティブサマリー")
        
        # 結果構築
        result = StrategicProposal(
            company_code=company_code,
            strategic_direction=growth_data.get("strategic_direction", ""),
            vision=growth_data.get("vision", ""),
            growth_strategies=growth_data.get("growth_strategies", []),
            dx_measures=dx_data.get("dx_measures", []),
            gx_measures=gx_data.get("gx_measures", []),
            hr_measures=hr_data.get("hr_measures", []),
            short_term_actions=action_data.get("short_term_actions", []),
            medium_term_actions=action_data.get("medium_term_actions", []),
            long_term_actions=action_data.get("long_term_actions", []),
            executive_summary=summary.strip()
        )
        
        print(f"[INFO] 戦略策定完了: {company_code}")
        return result
    
    def plan_all_companies(self, company_codes: list[str]) -> dict[str, StrategicProposal]:
        """全企業の戦略を策定"""
        results = {}
        
        for i, code in enumerate(company_codes, 1):
            print(f"\n[{i}/{len(company_codes)}] {code}")
            try:
                result = self.plan_strategy(code)
                results[code] = result
            except Exception as e:
                print(f"[ERROR] {code}: {e}")
                results[code] = StrategicProposal(company_code=code)
            
            if i < len(company_codes):
                time.sleep(2)
        
        return results


# =============================================================================
# 結果出力
# =============================================================================
def save_strategy_results(results: dict[str, StrategicProposal]):
    """戦略策定結果を保存"""
    
    # JSON形式
    all_results = {code: asdict(result) for code, result in results.items()}
    json_path = STRATEGY_DIR / "strategy_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {json_path}")
    
    # 企業別テキストレポート
    for code, result in results.items():
        report = generate_strategy_report(result)
        report_path = STRATEGY_DIR / f"{code}_strategy.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
    print(f"[SAVE] 企業別戦略: {len(results)}社")


def generate_strategy_report(result: StrategicProposal) -> str:
    """戦略レポートを生成"""
    report = f"""
================================================================================
戦略提案レポート: {result.company_code}
================================================================================

【エグゼクティブサマリー】
{result.executive_summary}

================================================================================
【戦略の方向性】
================================================================================
{result.strategic_direction}

【目指す姿（3-5年後）】
{result.vision}

================================================================================
【成長戦略】
================================================================================
"""
    
    for i, strategy in enumerate(result.growth_strategies, 1):
        if isinstance(strategy, dict):
            report += f"""
{i}. {strategy.get('name', '戦略')}
   概要: {strategy.get('description', '')}
   期待効果: {strategy.get('expected_effect', '')}
   優先度: {strategy.get('priority', '')}
"""
    
    report += """
================================================================================
【DX施策】
================================================================================
"""
    
    for i, measure in enumerate(result.dx_measures, 1):
        if isinstance(measure, dict):
            report += f"""
{i}. {measure.get('name', '施策')} [{measure.get('category', '')}]
   概要: {measure.get('description', '')}
   投資規模: {measure.get('investment', '')}
   導入期間: {measure.get('timeline', '')}
   期待効果: {measure.get('expected_effect', '')}
"""
    
    report += """
================================================================================
【GX施策】
================================================================================
"""
    
    for i, measure in enumerate(result.gx_measures, 1):
        if isinstance(measure, dict):
            report += f"""
{i}. {measure.get('name', '施策')} [{measure.get('category', '')}]
   概要: {measure.get('description', '')}
   CO2削減効果: {measure.get('co2_reduction', '')}
   期待効果: {measure.get('expected_effect', '')}
"""
    
    report += """
================================================================================
【人材施策】
================================================================================
"""
    
    for i, measure in enumerate(result.hr_measures, 1):
        if isinstance(measure, dict):
            report += f"""
{i}. {measure.get('name', '施策')} [{measure.get('category', '')}]
   概要: {measure.get('description', '')}
   対象: {measure.get('target', '')}
   期待効果: {measure.get('expected_effect', '')}
"""
    
    report += """
================================================================================
【実行ロードマップ】
================================================================================

■ 短期アクション（1年以内）
"""
    for action in result.short_term_actions:
        if isinstance(action, dict):
            report += f"- [{action.get('category', '')}] {action.get('action', '')}\n"
            report += f"  KPI: {action.get('kpi', '')}\n"
    
    report += """
■ 中期アクション（1-3年）
"""
    for action in result.medium_term_actions:
        if isinstance(action, dict):
            report += f"- [{action.get('category', '')}] {action.get('action', '')}\n"
            report += f"  KPI: {action.get('kpi', '')}\n"
    
    report += """
■ 長期アクション（3-5年）
"""
    for action in result.long_term_actions:
        if isinstance(action, dict):
            report += f"- [{action.get('category', '')}] {action.get('action', '')}\n"
            report += f"  KPI: {action.get('kpi', '')}\n"
    
    return report


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print("=" * 60)
    print("Step 6: 戦略策定・提案")
    print("=" * 60 + "\n")
    
    # 企業コードの取得
    company_master = load_company_master()
    company_codes = sorted(company_master.keys())
    print(f"[INFO] 対象企業: {len(company_codes)}社")
    
    # SWOT分析結果の確認
    swot_data = load_swot_results()
    if not swot_data:
        print("[WARNING] SWOT分析結果がありません。Step 5を先に実行してください。")
    
    # API キー確認
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n[ERROR] GOOGLE_API_KEY が設定されていません。")
        return None
    
    # 戦略策定
    planner = StrategyPlanner(api_key)
    results = planner.plan_all_companies(company_codes)
    
    # 結果保存
    save_strategy_results(results)
    
    # プロンプトログ保存
    log_text = planner.prompt_logger.export_to_text()
    log_path = PROMPT_LOG_DIR / "step6_prompt_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)
    print(f"[SAVE] {log_path}")
    
    print("\n" + "=" * 60)
    print("Step 6 完了！")
    print(f"出力先: {STRATEGY_DIR}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

