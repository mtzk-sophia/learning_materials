"""
Step 5: SWOT分析・課題抽出
==========================
- 財務分析 + 有報分析 + 外部情報を統合
- SWOT分析の実施
- クロスSWOT分析
- 経営課題の優先順位付け
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
EXTERNAL_DIR = DATA_DIR / "external"  # 外部情報（Step 4で収集）

# 出力ディレクトリ
SWOT_DIR = ANALYSIS_DIR / "swot"
SWOT_DIR.mkdir(parents=True, exist_ok=True)

# プロンプトログ
PROMPT_LOG_DIR = BASE_DIR / "output" / "prompt_logs"
PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_MODEL = "gemini-3-pro-preview"


# =============================================================================
# データクラス
# =============================================================================
@dataclass
class SWOTAnalysis:
    """SWOT分析結果"""
    company_code: str
    
    # SWOT要素
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    opportunities: list[str] = field(default_factory=list)
    threats: list[str] = field(default_factory=list)
    
    # クロスSWOT戦略
    so_strategies: list[str] = field(default_factory=list)  # 強み×機会
    wo_strategies: list[str] = field(default_factory=list)  # 弱み×機会
    st_strategies: list[str] = field(default_factory=list)  # 強み×脅威
    wt_strategies: list[str] = field(default_factory=list)  # 弱み×脅威
    
    # 優先課題
    priority_issues: list[dict] = field(default_factory=list)
    
    # 分析サマリー
    summary: str = ""


@dataclass
class PromptLogEntry:
    """プロンプトログエントリ"""
    timestamp: str
    company_code: str
    purpose: str
    input_prompt: str
    output_response: str
    model: str = GEMINI_MODEL


class PromptLogger:
    """プロンプトログ管理"""
    
    def __init__(self, log_name: str = "step5"):
        self.logs: list[PromptLogEntry] = []
        self.log_file = PROMPT_LOG_DIR / f"prompt_log_{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def add_log(self, entry: PromptLogEntry):
        self.logs.append(entry)
        self._save_logs()
    
    def _save_logs(self):
        logs_dict = [asdict(log) for log in self.logs]
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs_dict, f, ensure_ascii=False, indent=2)
    
    def export_to_text(self) -> str:
        text = "=" * 80 + "\n"
        text += "Step 5: SWOT分析 プロンプトログ\n"
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
def load_financial_analysis() -> dict:
    """財務分析結果を読み込み"""
    json_path = ANALYSIS_DIR / "reports" / "analysis_summary_for_ai.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_securities_analysis() -> dict:
    """有報分析結果を読み込み"""
    json_path = ANALYSIS_DIR / "llm_analysis" / "securities_report_analysis.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_external_info_for_company(location: str) -> str:
    """外部情報を読み込み（企業所在地に応じた情報を抽出）"""
    try:
        from external_info_loader import get_external_context_for_company
        return get_external_context_for_company(location)
    except ImportError:
        return ""


def load_company_master() -> dict:
    """企業マスタを読み込み"""
    json_path = PROCESSED_DIR / "company_master.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            masters = json.load(f)
            return {m["code"]: m for m in masters}
    return {}


# =============================================================================
# SWOT分析クラス
# =============================================================================
class SWOTAnalyzer:
    """SWOT分析を実行するクラス"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY が設定されていません。")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.prompt_logger = PromptLogger("step5")
        
        # データ読み込み
        self.financial_data = load_financial_analysis()
        self.securities_data = load_securities_analysis()
        self.company_master = load_company_master()
    
    def _call_gemini(self, prompt: str, company_code: str, purpose: str) -> str:
        """Gemini APIを呼び出す"""
        try:
            response = self.model.generate_content(prompt)
            result = response.text
            
            log_entry = PromptLogEntry(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                company_code=company_code,
                purpose=purpose,
                input_prompt=prompt,
                output_response=result,
            )
            self.prompt_logger.add_log(log_entry)
            
            return result
        except Exception as e:
            print(f"[ERROR] API呼び出しエラー: {e}")
            time.sleep(2)
            return ""
    
    def _parse_json_response(self, response: str) -> dict:
        """JSONレスポンスを解析"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {}
    
    def _prepare_company_context(self, company_code: str) -> str:
        """企業コンテキストを準備"""
        context_parts = []
        
        # 企業マスタ情報
        master = self.company_master.get(company_code, {})
        if master:
            context_parts.append(f"""
【企業基本情報】
- 企業コード: {company_code}
- 所在地: {master.get('location', '不明')}
- 業種: {master.get('industry', '不明')}
- 従業員数: {master.get('employees', '不明')}名
- 資本金: {master.get('capital', '不明')}億円
- 市場: {master.get('market', '不明')}
""")
        
        # 財務分析情報
        financial = self.financial_data.get(company_code, {})
        if financial:
            context_parts.append(f"""
【財務分析結果】
- 業績推移: {json.dumps(financial.get('業績推移', {}), ensure_ascii=False)}
- 収益性指標: {json.dumps(financial.get('収益性指標', {}), ensure_ascii=False)}
- 安全性指標: {json.dumps(financial.get('安全性指標', {}), ensure_ascii=False)}
- キャッシュフロー: {json.dumps(financial.get('キャッシュフロー_億円', {}), ensure_ascii=False)}
- 強み: {financial.get('分析結果', {}).get('強み', [])}
- 課題: {financial.get('分析結果', {}).get('課題', [])}
""")
        
        # 有報分析情報
        securities = self.securities_data.get(company_code, {})
        if securities:
            context_parts.append(f"""
【有価証券報告書分析】
- 事業セグメント: {securities.get('business_segments', [])}
- 経営理念: {securities.get('management_philosophy', '')}
- 重点戦略: {securities.get('key_strategies', [])}
- 強み: {securities.get('strengths', [])}
- 課題: {securities.get('challenges', [])}
- DX取組: {securities.get('dx_initiatives', [])}
- GX取組: {securities.get('gx_initiatives', [])}
- リスク: {securities.get('business_risks', [])}

【事業実績・計画】
- 中期経営計画の目標数値: {securities.get('medium_term_plan_targets', '情報なし')}
- セグメント別業績: {securities.get('segment_performance', '情報なし')}
- 受注・販売実績: {securities.get('order_sales_record', '情報なし')}

【経営者の分析（インサイト）】
{securities.get('management_analysis', '情報なし')}

【設備投資の戦略分析】
{securities.get('capex_plan', '情報なし')}
""")
        
        # 外部情報（テキストレポートから所在地に応じた情報を抽出）
        location = master.get('location', '')
        external_context = load_external_info_for_company(location)
        if external_context:
            context_parts.append(f"""
【外部環境情報（業界レポートより抽出）】
{external_context}
""")
        
        return "\n".join(context_parts)
    
    def analyze_swot(self, company_code: str) -> SWOTAnalysis:
        """SWOT分析を実行"""
        print(f"[INFO] SWOT分析開始: {company_code}")
        
        context = self._prepare_company_context(company_code)
        
        # SWOT要素の抽出
        swot_prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下の企業情報をもとに、SWOT分析を実施してください。

{context}

【分析の観点】
建設業界特有の視点を含めて分析してください：
- 2024年問題（時間外労働規制）への対応
- DX/GX（デジタル・グリーントランスフォーメーション）
- 人材確保・技術承継
- 地域特性と市場環境
- 官公庁/民間のバランス

【出力形式】
JSON形式で出力してください。各項目は3〜5つ程度、具体的に記載してください。
```json
{{
    "strengths": ["強み1（根拠を含めて）", "強み2", ...],
    "weaknesses": ["弱み1（根拠を含めて）", "弱み2", ...],
    "opportunities": ["機会1（市場環境・トレンドに基づく）", "機会2", ...],
    "threats": ["脅威1（外部環境リスク）", "脅威2", ...]
}}
```
"""
        
        swot_response = self._call_gemini(swot_prompt, company_code, "SWOT分析")
        swot_data = self._parse_json_response(swot_response)
        
        time.sleep(5)
        
        # クロスSWOT戦略の策定
        cross_swot_prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下のSWOT分析結果をもとに、クロスSWOT分析を行い、戦略オプションを策定してください。

【SWOT分析結果】
- 強み（S）: {swot_data.get('strengths', [])}
- 弱み（W）: {swot_data.get('weaknesses', [])}
- 機会（O）: {swot_data.get('opportunities', [])}
- 脅威（T）: {swot_data.get('threats', [])}

【クロスSWOT戦略】
1. SO戦略（強み×機会）: 強みを活かして機会を最大化する攻めの戦略
2. WO戦略（弱み×機会）: 弱みを克服して機会を活かす改善戦略
3. ST戦略（強み×脅威）: 強みを活かして脅威を回避・軽減する防衛戦略
4. WT戦略（弱み×脅威）: 弱みと脅威による最悪シナリオを回避する戦略

【出力形式】
JSON形式で出力してください。各戦略は2〜3つ、具体的なアクションを含めてください。
```json
{{
    "so_strategies": ["SO戦略1：具体的なアクション", "SO戦略2", ...],
    "wo_strategies": ["WO戦略1：具体的なアクション", "WO戦略2", ...],
    "st_strategies": ["ST戦略1：具体的なアクション", "ST戦略2", ...],
    "wt_strategies": ["WT戦略1：具体的なアクション", "WT戦略2", ...]
}}
```
"""
        
        cross_response = self._call_gemini(cross_swot_prompt, company_code, "クロスSWOT分析")
        cross_data = self._parse_json_response(cross_response)
        
        time.sleep(5)
        
        # 優先課題の特定
        priority_prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下の分析結果から、この企業が優先的に取り組むべき経営課題を特定してください。

【SWOT分析結果】
{json.dumps(swot_data, ensure_ascii=False, indent=2)}

【クロスSWOT戦略】
{json.dumps(cross_data, ensure_ascii=False, indent=2)}

【課題特定の観点】
1. 緊急度（短期/中期/長期）
2. 重要度（経営への影響度）
3. 実現可能性
4. 建設業界の構造変化への対応

【出力形式】
JSON形式で出力してください。優先度順に5つ程度の課題を挙げてください。
```json
{{
    "priority_issues": [
        {{
            "issue": "課題の内容",
            "urgency": "高/中/低",
            "importance": "高/中/低",
            "timeframe": "短期（1年以内）/中期（1-3年）/長期（3-5年）",
            "rationale": "この課題を優先すべき理由",
            "related_strategy": "関連するクロスSWOT戦略"
        }}
    ]
}}
```
"""
        
        priority_response = self._call_gemini(priority_prompt, company_code, "優先課題特定")
        priority_data = self._parse_json_response(priority_response)
        
        time.sleep(5)
        
        # サマリー生成
        summary_prompt = f"""以下のSWOT分析結果を300字程度でサマリーしてください。
企業の現状認識と今後の方向性がわかるように簡潔にまとめてください。

【SWOT分析】
{json.dumps(swot_data, ensure_ascii=False)}

【優先課題】
{json.dumps(priority_data.get('priority_issues', [])[:3], ensure_ascii=False)}

サマリーテキストのみを出力してください。
"""
        
        summary = self._call_gemini(summary_prompt, company_code, "サマリー生成")
        
        # 結果の構築
        result = SWOTAnalysis(
            company_code=company_code,
            strengths=swot_data.get("strengths", []),
            weaknesses=swot_data.get("weaknesses", []),
            opportunities=swot_data.get("opportunities", []),
            threats=swot_data.get("threats", []),
            so_strategies=cross_data.get("so_strategies", []),
            wo_strategies=cross_data.get("wo_strategies", []),
            st_strategies=cross_data.get("st_strategies", []),
            wt_strategies=cross_data.get("wt_strategies", []),
            priority_issues=priority_data.get("priority_issues", []),
            summary=summary.strip()
        )
        
        print(f"[INFO] SWOT分析完了: {company_code}")
        return result
    
    def analyze_all_companies(self, company_codes: list[str]) -> dict[str, SWOTAnalysis]:
        """全企業のSWOT分析を実行"""
        results = {}
        
        for i, code in enumerate(company_codes, 1):
            print(f"\n[{i}/{len(company_codes)}] {code}")
            try:
                result = self.analyze_swot(code)
                results[code] = result
            except Exception as e:
                print(f"[ERROR] {code}: {e}")
                results[code] = SWOTAnalysis(company_code=code)
            
            if i < len(company_codes):
                time.sleep(100)
        
        return results


# =============================================================================
# 結果出力
# =============================================================================
def save_swot_results(results: dict[str, SWOTAnalysis]):
    """SWOT分析結果を保存"""
    
    # JSON形式
    all_results = {code: asdict(result) for code, result in results.items()}
    json_path = SWOT_DIR / "swot_analysis_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {json_path}")
    
    # 企業別テキストレポート
    for code, result in results.items():
        report = generate_swot_report(result)
        report_path = SWOT_DIR / f"{code}_swot.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
    print(f"[SAVE] 企業別SWOT: {len(results)}社")


def generate_swot_report(result: SWOTAnalysis) -> str:
    """SWOTレポートを生成"""
    report = f"""
================================================================================
SWOT分析レポート: {result.company_code}
================================================================================

【サマリー】
{result.summary}

================================================================================
【SWOT分析】
================================================================================

■ 強み（Strengths）
{chr(10).join('- ' + s for s in result.strengths) if result.strengths else '- 情報なし'}

■ 弱み（Weaknesses）
{chr(10).join('- ' + w for w in result.weaknesses) if result.weaknesses else '- 情報なし'}

■ 機会（Opportunities）
{chr(10).join('- ' + o for o in result.opportunities) if result.opportunities else '- 情報なし'}

■ 脅威（Threats）
{chr(10).join('- ' + t for t in result.threats) if result.threats else '- 情報なし'}

================================================================================
【クロスSWOT戦略】
================================================================================

■ SO戦略（強み×機会）: 攻めの戦略
{chr(10).join('- ' + s for s in result.so_strategies) if result.so_strategies else '- 情報なし'}

■ WO戦略（弱み×機会）: 改善戦略
{chr(10).join('- ' + s for s in result.wo_strategies) if result.wo_strategies else '- 情報なし'}

■ ST戦略（強み×脅威）: 防衛戦略
{chr(10).join('- ' + s for s in result.st_strategies) if result.st_strategies else '- 情報なし'}

■ WT戦略（弱み×脅威）: 回避戦略
{chr(10).join('- ' + s for s in result.wt_strategies) if result.wt_strategies else '- 情報なし'}

================================================================================
【優先課題】
================================================================================
"""
    
    for i, issue in enumerate(result.priority_issues, 1):
        if isinstance(issue, dict):
            report += f"""
{i}. {issue.get('issue', '課題')}
   - 緊急度: {issue.get('urgency', '-')}
   - 重要度: {issue.get('importance', '-')}
   - 時間軸: {issue.get('timeframe', '-')}
   - 理由: {issue.get('rationale', '-')}
"""
    
    return report


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print("=" * 60)
    print("Step 5: SWOT分析・課題抽出")
    print("=" * 60 + "\n")
    
    # 企業コードの取得
    company_master = load_company_master()
    company_codes = sorted(company_master.keys())
    print(f"[INFO] 分析対象: {len(company_codes)}社")
    
    # API キー確認
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n[ERROR] GOOGLE_API_KEY が設定されていません。")
        return None
    
    # SWOT分析実行
    analyzer = SWOTAnalyzer(api_key)
    results = analyzer.analyze_all_companies(company_codes)
    
    # 結果保存
    save_swot_results(results)
    
    # プロンプトログ保存
    log_text = analyzer.prompt_logger.export_to_text()
    log_path = PROMPT_LOG_DIR / "step5_prompt_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)
    print(f"[SAVE] {log_path}")
    
    print("\n" + "=" * 60)
    print("Step 5 完了！")
    print(f"出力先: {SWOT_DIR}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

