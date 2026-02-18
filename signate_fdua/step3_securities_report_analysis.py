"""
Step 3: 有価証券報告書分析（LLM活用）
=====================================
- Gemini 1.5 Flash を使用した有報テキスト分析
- 事業構造の理解
- 経営方針の把握
- リスク要因の整理
- 強み・差別化要因の抽出
- プロンプトログの記録
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
REPORTS_TEXT_DIR = PROCESSED_DIR / "securities_reports_text"

# 出力ディレクトリ
LLM_ANALYSIS_DIR = ANALYSIS_DIR / "llm_analysis"
LLM_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# プロンプトログディレクトリ
PROMPT_LOG_DIR = BASE_DIR / "output" / "prompt_logs"
PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Gemini API設定
# 環境変数 GOOGLE_API_KEY を設定してください
GEMINI_MODEL = "gemini-3-pro-preview"


# =============================================================================
# プロンプトログ管理
# =============================================================================
@dataclass
class PromptLogEntry:
    """プロンプトログエントリ"""
    timestamp: str
    company_code: str
    purpose: str
    input_prompt: str
    output_response: str
    model: str = GEMINI_MODEL
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None


class PromptLogger:
    """プロンプトログを管理するクラス"""
    
    def __init__(self):
        self.logs: list[PromptLogEntry] = []
        self.log_file = PROMPT_LOG_DIR / f"prompt_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def add_log(self, entry: PromptLogEntry):
        """ログエントリを追加"""
        self.logs.append(entry)
        self._save_logs()
    
    def _save_logs(self):
        """ログをJSONファイルに保存"""
        logs_dict = [asdict(log) for log in self.logs]
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs_dict, f, ensure_ascii=False, indent=2)
    
    def export_to_text(self) -> str:
        """テキスト形式でエクスポート（提出用）"""
        text = "=" * 80 + "\n"
        text += "プロンプトログ（Prompt Log）\n"
        text += f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"使用モデル: {GEMINI_MODEL}\n"
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
    
    def save_text_log(self):
        """テキスト形式のログを保存"""
        text = self.export_to_text()
        text_file = PROMPT_LOG_DIR / "prompt_log.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[SAVE] {text_file}")


# =============================================================================
# 有価証券報告書分析結果
# =============================================================================
@dataclass
class SecuritiesReportAnalysis:
    """有価証券報告書分析結果"""
    company_code: str
    company_name: str
    
    # 事業構造
    business_segments: list[str] = field(default_factory=list)
    main_business_description: str = ""
    sales_structure: dict = field(default_factory=dict)  # 官公庁/民間、元請/下請 等
    
    # 経営方針
    management_philosophy: str = ""
    medium_term_plan: str = ""
    key_strategies: list[str] = field(default_factory=list)
    kpis: list[str] = field(default_factory=list)
    
    # リスク情報
    business_risks: list[str] = field(default_factory=list)
    financial_risks: list[str] = field(default_factory=list)
    external_risks: list[str] = field(default_factory=list)
    
    # 強み・差別化
    strengths: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    
    # 課題・対処方針
    challenges: list[str] = field(default_factory=list)
    countermeasures: list[str] = field(default_factory=list)
    
    # DX/GX対応
    dx_initiatives: list[str] = field(default_factory=list)
    gx_initiatives: list[str] = field(default_factory=list)
    
    # 人材関連
    hr_strategy: str = ""
    employee_info: dict = field(default_factory=dict)
    
    # サステナビリティ
    sustainability_initiatives: list[str] = field(default_factory=list)
    
    # ★ Step 1で追加された新フィールド（構造化データから直接取得）
    medium_term_plan_targets: str = ""  # 中期経営計画の目標数値
    segment_performance: str = ""  # セグメント別業績
    order_sales_record: str = ""  # 受注・販売実績
    management_analysis: str = ""  # 経営者の分析
    capex_plan: str = ""  # 設備投資計画
    
    # 分析サマリー
    summary: str = ""


# =============================================================================
# Gemini API クライアント
# =============================================================================
class GeminiAnalyzer:
    """Gemini APIを使用した分析クラス"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY が設定されていません。\n"
                "環境変数に設定するか、コンストラクタに渡してください。\n"
                "例: export GOOGLE_API_KEY='your-api-key'"
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.prompt_logger = PromptLogger()
    
    def _call_gemini(
        self, 
        prompt: str, 
        company_code: str, 
        purpose: str,
        max_retries: int = 3
    ) -> str:
        """Gemini APIを呼び出す"""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                result = response.text
                
                # ログ記録
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
                print(f"[WARNING] API呼び出しエラー (試行 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数バックオフ
                else:
                    raise
        
        return ""
    
    def analyze_business_structure(self, text: str, company_code: str) -> dict:
        """事業構造を分析"""
        prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下の有価証券報告書のテキストから、事業構造に関する情報を抽出してください。

【抽出項目】
1. 事業セグメント（土木、建築、不動産等）
2. 主要な事業内容の説明
3. 販路構成（官公庁/民間の比率、元請/下請の状況）
4. 地域展開（主要な営業エリア）
5. 主要な顧客層

【出力形式】
JSON形式で出力してください。
```json
{{
    "business_segments": ["セグメント1", "セグメント2"],
    "main_business_description": "事業内容の説明",
    "sales_structure": {{
        "public_private_ratio": "官公庁:民間の比率",
        "prime_sub_ratio": "元請:下請の比率",
        "main_regions": ["地域1", "地域2"],
        "main_customers": ["顧客層1", "顧客層2"]
    }}
}}
```

【有価証券報告書テキスト】
{text}
"""
        response = self._call_gemini(prompt, company_code, "事業構造分析")
        return self._parse_json_response(response)
    
    def analyze_management_policy(self, text: str, company_code: str) -> dict:
        """経営方針を分析"""
        prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下の有価証券報告書のテキストから、経営方針に関する情報を抽出してください。

【抽出項目】
1. 経営理念・ビジョン
2. 中期経営計画の概要（名称、期間、重点テーマ）
3. 重点戦略・施策（3〜5つ）
4. 経営目標（KPI）

【出力形式】
JSON形式で出力してください。
```json
{{
    "management_philosophy": "経営理念",
    "medium_term_plan": "中期経営計画の概要",
    "key_strategies": ["戦略1", "戦略2", "戦略3"],
    "kpis": ["KPI1", "KPI2"]
}}
```

【有価証券報告書テキスト】
{text[:15000]}
"""
        response = self._call_gemini(prompt, company_code, "経営方針分析")
        return self._parse_json_response(response)
    
    def analyze_risks(self, text: str, company_code: str) -> dict:
        """リスク情報を分析"""
        prompt = f"""あなたは建設業界に精通したリスクアナリストです。
以下の有価証券報告書のテキストから、リスク情報を抽出してください。

【抽出項目】
1. 事業リスク（人材不足、工事品質、競合等）
2. 財務リスク（資金繰り、与信、原価高騰等）
3. 外部環境リスク（市況変動、法規制、自然災害等）

各リスクについて、具体的な内容と会社が認識している対策も含めてください。

【出力形式】
JSON形式で出力してください。
```json
{{
    "business_risks": [
        {{"risk": "リスク内容", "countermeasure": "対策"}}
    ],
    "financial_risks": [
        {{"risk": "リスク内容", "countermeasure": "対策"}}
    ],
    "external_risks": [
        {{"risk": "リスク内容", "countermeasure": "対策"}}
    ]
}}
```

【有価証券報告書テキスト】
{text}
"""
        response = self._call_gemini(prompt, company_code, "リスク分析")
        return self._parse_json_response(response)
    
    def analyze_strengths_and_challenges(self, text: str, company_code: str) -> dict:
        """強み・課題を分析"""
        prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下の有価証券報告書のテキストから、企業の強み・差別化要因と課題を抽出してください。

【抽出項目】
1. 強み・差別化要因
   - 技術力（保有技術、特許、ノウハウ等）
   - 資格・認証（ISO、施工実績等）
   - その他の競争優位性

2. 経営課題と対処方針
   - 認識している課題
   - 具体的な対処方針

3. DX（デジタルトランスフォーメーション）への取り組み
   - ICT施工、BIM/CIM、業務効率化等

4. GX（グリーントランスフォーメーション）への取り組み
   - 環境配慮、脱炭素、ZEB/ZEH等

【出力形式】
JSON形式で出力してください。
```json
{{
    "strengths": ["強み1", "強み2"],
    "technologies": ["技術1", "技術2"],
    "certifications": ["資格・認証1", "資格・認証2"],
    "challenges": ["課題1", "課題2"],
    "countermeasures": ["対処方針1", "対処方針2"],
    "dx_initiatives": ["DX取組1", "DX取組2"],
    "gx_initiatives": ["GX取組1", "GX取組2"]
}}
```

【有価証券報告書テキスト】
{text}
"""
        response = self._call_gemini(prompt, company_code, "強み・課題分析")
        return self._parse_json_response(response)
    
    def analyze_hr_and_sustainability(self, text: str, company_code: str) -> dict:
        """人材・サステナビリティを分析"""
        prompt = f"""あなたは建設業界に精通した人事・ESGコンサルタントです。
以下の有価証券報告書のテキストから、人材戦略とサステナビリティに関する情報を抽出してください。

【抽出項目】
1. 人材戦略
   - 人材育成方針
   - 働き方改革への取り組み
   - 2024年問題への対応
   - ダイバーシティ推進

2. 従業員情報
   - 従業員数
   - 平均年齢
   - 女性管理職比率等

3. サステナビリティへの取り組み
   - ESG経営
   - SDGs対応
   - 環境目標

【出力形式】
JSON形式で出力してください。
```json
{{
    "hr_strategy": "人材戦略の概要",
    "employee_info": {{
        "total_employees": 数値,
        "average_age": 数値,
        "female_manager_ratio": "比率"
    }},
    "sustainability_initiatives": ["取組1", "取組2"]
}}
```

【有価証券報告書テキスト】
{text[:15000]}
"""
        response = self._call_gemini(prompt, company_code, "人材・サステナビリティ分析")
        return self._parse_json_response(response)
    
    def analyze_management_insights(self, text: str, company_code: str) -> dict:
        """経営者の分析から提案書に活用できるインサイトを抽出"""
        prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下は有価証券報告書における「経営者の視点による分析」のテキストです。
提案書作成に活用するため、経営者の認識を構造化して抽出してください。

【抽出の観点】
1. 増減益の要因分析: なぜ業績が改善/悪化したのか、経営者が認識している主因を特定
2. 経営者の課題認識: 今後の経営で最も重要と認識している課題は何か
3. 今後の見通し: 経営者が次期以降をどう見通しているか（楽観/慎重/懸念等）
4. 提案のヒント: この企業に対する戦略提案で活用できるポイント
   （例：経営者が認識しているが対策が不十分な領域、言及していない重要課題等）

【出力形式】
```json
{{
    "revenue_change_factors": ["増減収の主因1", "主因2"],
    "profit_change_factors": ["増減益の主因1", "主因2"],
    "management_concerns": ["経営者が認識している懸念事項1", "懸念事項2"],
    "future_outlook": "経営者の今後の見通し（1-2文）",
    "proposal_hints": ["戦略提案で活用すべきポイント1", "ポイント2", "ポイント3"],
    "unaddressed_gaps": ["経営者が言及していないが重要と思われる課題1", "課題2"]
}}
```

【テキスト】
{text}
"""
        response = self._call_gemini(prompt, company_code, "経営者分析インサイト抽出")
        return self._parse_json_response(response)

    def analyze_capex_strategy(self, capex_text: str, current_capex_text: str, company_code: str) -> dict:
        """設備投資計画の戦略的意図を分析"""
        prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下は有価証券報告書における「設備投資の実績」と「設備投資計画」のテキストです。
投資の戦略的意図を分析し、提案書で活用できる形で構造化してください。

【分析の観点】
1. 投資分類: 各投資項目をDX投資/GX投資/成長投資/維持更新投資に分類
2. 戦略との整合: 中期経営計画や経営課題との関連を評価
3. 不足している投資: 建設業界のトレンド（i-Construction、BIM/CIM、ZEB等）を踏まえ、この企業に不足していると思われる投資領域を特定
4. 提案の方向性: 追加で提案すべき設備投資の方向性

【出力形式】
```json
{{
    "investment_classification": [
        {{"item": "投資項目名", "category": "DX/GX/成長/維持更新", "amount": "金額", "strategic_purpose": "戦略的意図"}}
    ],
    "dx_investment_level": "積極的/標準的/不十分",
    "gx_investment_level": "積極的/標準的/不十分",
    "missing_investments": ["不足している投資領域1", "不足している投資領域2"],
    "proposal_directions": ["追加提案の方向性1", "追加提案の方向性2"]
}}
```

【設備投資の実績】
{current_capex_text}

【設備投資計画（翌期以降）】
{capex_text}
"""
        response = self._call_gemini(prompt, company_code, "設備投資戦略分析")
        return self._parse_json_response(response)

    def generate_summary(self, analysis_result: SecuritiesReportAnalysis) -> str:
        """分析結果のサマリーを生成"""
        prompt = f"""あなたは建設業界に精通した経営コンサルタントです。
以下の有価証券報告書分析結果をもとに、企業の概要サマリーを作成してください。

【分析結果】
- 企業コード: {analysis_result.company_code}
- 事業セグメント: {analysis_result.business_segments}
- 経営理念: {analysis_result.management_philosophy}
- 重点戦略: {analysis_result.key_strategies}
- 強み: {analysis_result.strengths}
- 課題: {analysis_result.challenges}
- DX取組: {analysis_result.dx_initiatives}
- GX取組: {analysis_result.gx_initiatives}

【要求】
1. 300〜500字程度で簡潔にまとめてください
2. 事業の特徴、強み、課題、今後の方向性を含めてください
3. 建設業界の文脈で重要なポイントを強調してください

【出力】
サマリーテキストのみを出力してください（JSON形式不要）。
"""
        response = self._call_gemini(
            prompt, 
            analysis_result.company_code, 
            "サマリー生成"
        )
        return response.strip()
    
    def _parse_json_response(self, response: str) -> dict:
        """APIレスポンスからJSONを抽出"""
        try:
            # マークダウンのコードブロックを除去
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON解析エラー: {e}")
            return {}
    
    def analyze_company(self, company_code: str) -> SecuritiesReportAnalysis:
        """1社分の有価証券報告書を分析"""
        print(f"[INFO] 分析開始: {company_code}")
        
        # テキストファイルの読み込み
        full_text_path = REPORTS_TEXT_DIR / f"{company_code}_full_text.txt"
        sections_path = REPORTS_TEXT_DIR / f"{company_code}_sections.json"
        
        if not full_text_path.exists():
            print(f"[WARNING] ファイルが見つかりません: {full_text_path}")
            return SecuritiesReportAnalysis(company_code=company_code, company_name="")
        
        with open(full_text_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        
        with open(sections_path, "r", encoding="utf-8") as f:
            sections = json.load(f)
        
        # 各分析を実行
        result = SecuritiesReportAnalysis(
            company_code=company_code,
            company_name=""
        )
        
        # 1. 事業構造分析
        print(f"  - 事業構造分析中...")
        business_text = sections.get("事業内容", "") + sections.get("その他", "")
        business_data = self.analyze_business_structure(business_text, company_code)
        if business_data:
            result.business_segments = business_data.get("business_segments", [])
            result.main_business_description = business_data.get("main_business_description", "")
            result.sales_structure = business_data.get("sales_structure", {})
        
        time.sleep(1)  # API制限対策
        
        # 2. 経営方針分析
        print(f"  - 経営方針分析中...")
        policy_text = sections.get("経営方針", "")
        policy_data = self.analyze_management_policy(policy_text, company_code)
        if policy_data:
            result.management_philosophy = policy_data.get("management_philosophy", "")
            result.medium_term_plan = policy_data.get("medium_term_plan", "")
            result.key_strategies = policy_data.get("key_strategies", [])
            result.kpis = policy_data.get("kpis", [])
        
        time.sleep(1)
        
        # 3. リスク分析
        print(f"  - リスク分析中...")
        risk_text = sections.get("リスク情報", "")
        risk_data = self.analyze_risks(risk_text, company_code)
        if risk_data:
            result.business_risks = [r.get("risk", "") for r in risk_data.get("business_risks", [])]
            result.financial_risks = [r.get("risk", "") for r in risk_data.get("financial_risks", [])]
            result.external_risks = [r.get("risk", "") for r in risk_data.get("external_risks", [])]
        
        time.sleep(1)
        
        # 4. 強み・課題分析
        print(f"  - 強み・課題分析中...")
        challenges_text = sections.get("経営上の課題", "") + sections.get("研究開発", "")
        strengths_data = self.analyze_strengths_and_challenges(challenges_text, company_code)
        if strengths_data:
            result.strengths = strengths_data.get("strengths", [])
            result.technologies = strengths_data.get("technologies", [])
            result.certifications = strengths_data.get("certifications", [])
            result.challenges = strengths_data.get("challenges", [])
            result.countermeasures = strengths_data.get("countermeasures", [])
            result.dx_initiatives = strengths_data.get("dx_initiatives", [])
            result.gx_initiatives = strengths_data.get("gx_initiatives", [])
        
        time.sleep(1)
        
        # 5. 人材・サステナビリティ分析
        print(f"  - 人材・サステナビリティ分析中...")
        hr_text = sections.get("従業員の状況", "") + sections.get("経営方針", "")
        hr_data = self.analyze_hr_and_sustainability(hr_text, company_code)
        if hr_data:
            result.hr_strategy = hr_data.get("hr_strategy", "")
            result.employee_info = hr_data.get("employee_info", {})
            result.sustainability_initiatives = hr_data.get("sustainability_initiatives", [])
        
        time.sleep(1)
        
        # 6. Step 1の構造化JSONから数値データを取得（LLM不要：正確性を保持）
        print(f"  - 構造化データ取得中...")
        result.medium_term_plan_targets = sections.get("中期経営計画_目標数値", "")
        result.segment_performance = sections.get("セグメント別業績", "")
        result.order_sales_record = sections.get("受注販売実績", "")
        
        time.sleep(1)
        
        # 7. 経営者の分析テキストをLLMで解釈（叙述データ → インサイト抽出）
        print(f"  - 経営者分析インサイト抽出中...")
        mgmt_text = sections.get("経営者の分析", "")
        if mgmt_text:
            mgmt_insights = self.analyze_management_insights(mgmt_text, company_code)
            if mgmt_insights:
                # インサイトをテキスト形式に統合して格納
                insight_parts = []
                for factor in mgmt_insights.get("revenue_change_factors", []):
                    insight_parts.append(f"[増減収要因] {factor}")
                for factor in mgmt_insights.get("profit_change_factors", []):
                    insight_parts.append(f"[増減益要因] {factor}")
                for concern in mgmt_insights.get("management_concerns", []):
                    insight_parts.append(f"[経営者の懸念] {concern}")
                outlook = mgmt_insights.get("future_outlook", "")
                if outlook:
                    insight_parts.append(f"[今後の見通し] {outlook}")
                for hint in mgmt_insights.get("proposal_hints", []):
                    insight_parts.append(f"[提案のヒント] {hint}")
                for gap in mgmt_insights.get("unaddressed_gaps", []):
                    insight_parts.append(f"[未対応の課題] {gap}")
                result.management_analysis = "\n".join(insight_parts)
            else:
                result.management_analysis = mgmt_text
        
        time.sleep(1)
        
        # 8. 設備投資計画のLLM分析（投資の戦略的意図を解釈）
        print(f"  - 設備投資戦略分析中...")
        capex_plan_text = sections.get("設備投資_計画", "")
        capex_actual_text = sections.get("設備投資_実績", "")
        if capex_plan_text or capex_actual_text:
            capex_insights = self.analyze_capex_strategy(
                capex_plan_text, capex_actual_text, company_code
            )
            if capex_insights:
                # インサイトをテキスト形式に統合して格納
                capex_parts = []
                for inv in capex_insights.get("investment_classification", []):
                    if isinstance(inv, dict):
                        capex_parts.append(
                            f"[{inv.get('category', '')}] {inv.get('item', '')}: "
                            f"{inv.get('amount', '')} - {inv.get('strategic_purpose', '')}"
                        )
                dx_level = capex_insights.get("dx_investment_level", "")
                gx_level = capex_insights.get("gx_investment_level", "")
                if dx_level:
                    capex_parts.append(f"[DX投資水準] {dx_level}")
                if gx_level:
                    capex_parts.append(f"[GX投資水準] {gx_level}")
                for missing in capex_insights.get("missing_investments", []):
                    capex_parts.append(f"[不足している投資] {missing}")
                for direction in capex_insights.get("proposal_directions", []):
                    capex_parts.append(f"[追加提案の方向性] {direction}")
                result.capex_plan = "\n".join(capex_parts)
            else:
                result.capex_plan = capex_plan_text
        
        # 9. サマリー生成
        print(f"  - サマリー生成中...")
        result.summary = self.generate_summary(result)
        
        print(f"[INFO] 分析完了: {company_code}")
        return result
    
    def analyze_all_companies(self, company_codes: list[str]) -> dict[str, SecuritiesReportAnalysis]:
        """全企業の分析を実行"""
        results = {}
        
        for i, code in enumerate(company_codes, 1):
            print(f"\n[{i}/{len(company_codes)}] 企業コード: {code}")
            try:
                result = self.analyze_company(code)
                results[code] = result
            except Exception as e:
                print(f"[ERROR] {code} の分析でエラー: {e}")
                results[code] = SecuritiesReportAnalysis(company_code=code, company_name="")
            
            # API制限対策
            if i < len(company_codes):
                time.sleep(2)
        
        return results


# =============================================================================
# 結果出力
# =============================================================================
def save_analysis_results(results: dict[str, SecuritiesReportAnalysis]):
    """分析結果を保存"""
    
    # 1. JSON形式で保存
    all_results = {}
    for code, result in results.items():
        all_results[code] = asdict(result)
    
    json_path = LLM_ANALYSIS_DIR / "securities_report_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {json_path}")
    
    # 2. 企業別テキストレポート
    for code, result in results.items():
        report = generate_text_report(result)
        report_path = LLM_ANALYSIS_DIR / f"{code}_securities_analysis.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
    print(f"[SAVE] 企業別レポート: {len(results)}社")
    
    # 3. 全社まとめレポート
    all_report = generate_all_companies_report(results)
    all_report_path = LLM_ANALYSIS_DIR / "all_securities_analysis.txt"
    with open(all_report_path, "w", encoding="utf-8") as f:
        f.write(all_report)
    print(f"[SAVE] {all_report_path}")


def generate_text_report(result: SecuritiesReportAnalysis) -> str:
    """テキスト形式のレポートを生成"""
    report = f"""
================================================================================
有価証券報告書分析レポート: {result.company_code}
================================================================================

【サマリー】
{result.summary}

--------------------------------------------------------------------------------
【事業構造】
■ 事業セグメント
{chr(10).join('- ' + s for s in result.business_segments) if result.business_segments else '- 情報なし'}

■ 事業内容
{result.main_business_description or '情報なし'}

■ 販路構成
{json.dumps(result.sales_structure, ensure_ascii=False, indent=2) if result.sales_structure else '情報なし'}

--------------------------------------------------------------------------------
【経営方針】
■ 経営理念
{result.management_philosophy or '情報なし'}

■ 中期経営計画
{result.medium_term_plan or '情報なし'}

■ 重点戦略
{chr(10).join('- ' + s for s in result.key_strategies) if result.key_strategies else '- 情報なし'}

■ KPI
{chr(10).join('- ' + s for s in result.kpis) if result.kpis else '- 情報なし'}

--------------------------------------------------------------------------------
【リスク情報】
■ 事業リスク
{chr(10).join('- ' + s for s in result.business_risks) if result.business_risks else '- 情報なし'}

■ 財務リスク
{chr(10).join('- ' + s for s in result.financial_risks) if result.financial_risks else '- 情報なし'}

■ 外部環境リスク
{chr(10).join('- ' + s for s in result.external_risks) if result.external_risks else '- 情報なし'}

--------------------------------------------------------------------------------
【強み・差別化】
■ 強み
{chr(10).join('- ' + s for s in result.strengths) if result.strengths else '- 情報なし'}

■ 保有技術
{chr(10).join('- ' + s for s in result.technologies) if result.technologies else '- 情報なし'}

■ 資格・認証
{chr(10).join('- ' + s for s in result.certifications) if result.certifications else '- 情報なし'}

--------------------------------------------------------------------------------
【課題・対処方針】
■ 経営課題
{chr(10).join('- ' + s for s in result.challenges) if result.challenges else '- 情報なし'}

■ 対処方針
{chr(10).join('- ' + s for s in result.countermeasures) if result.countermeasures else '- 情報なし'}

--------------------------------------------------------------------------------
【DX/GX対応】
■ DX（デジタルトランスフォーメーション）
{chr(10).join('- ' + s for s in result.dx_initiatives) if result.dx_initiatives else '- 情報なし'}

■ GX（グリーントランスフォーメーション）
{chr(10).join('- ' + s for s in result.gx_initiatives) if result.gx_initiatives else '- 情報なし'}

--------------------------------------------------------------------------------
【人材・サステナビリティ】
■ 人材戦略
{result.hr_strategy or '情報なし'}

■ 従業員情報
{json.dumps(result.employee_info, ensure_ascii=False, indent=2) if result.employee_info else '情報なし'}

■ サステナビリティ
{chr(10).join('- ' + s for s in result.sustainability_initiatives) if result.sustainability_initiatives else '- 情報なし'}

--------------------------------------------------------------------------------
【事業実績・計画（Step 1構造化データより）】
■ 中期経営計画の目標数値
{result.medium_term_plan_targets or '情報なし'}

■ セグメント別業績
{result.segment_performance or '情報なし'}

■ 受注・販売実績
{result.order_sales_record or '情報なし'}

■ 経営者の分析
{result.management_analysis or '情報なし'}

■ 設備投資計画
{result.capex_plan or '情報なし'}

================================================================================
"""
    return report


def generate_all_companies_report(results: dict[str, SecuritiesReportAnalysis]) -> str:
    """全社まとめレポートを生成"""
    report = "=" * 80 + "\n"
    report += "有価証券報告書分析レポート（全社まとめ）\n"
    report += f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += "=" * 80 + "\n\n"
    
    for code, result in results.items():
        report += generate_text_report(result)
        report += "\n\n"
    
    return report


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print("=" * 60)
    print("Step 3: 有価証券報告書分析（Gemini 2.5 Flash）")
    print("=" * 60 + "\n")
    
    # 企業コードの取得
    company_codes = []
    for file in REPORTS_TEXT_DIR.glob("*_full_text.txt"):
        code = file.stem.replace("_full_text", "")
        company_codes.append(code)
    
    company_codes.sort()
    print(f"[INFO] 分析対象企業: {len(company_codes)}社")
    print(f"       {company_codes}")
    
    # API キーの確認
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n[ERROR] GOOGLE_API_KEY が設定されていません。")
        print("以下のコマンドで環境変数を設定してください：")
        print("  export GOOGLE_API_KEY='your-api-key'")
        print("\nまたは、.envファイルを作成してください。")
        return None, None
    
    # 分析実行
    print("\n[Step 3.1] Gemini APIによる分析開始")
    analyzer = GeminiAnalyzer(api_key)
    results = analyzer.analyze_all_companies(company_codes)
    
    # 結果保存
    print("\n[Step 3.2] 分析結果の保存")
    save_analysis_results(results)
    
    # プロンプトログ保存
    print("\n[Step 3.3] プロンプトログの保存")
    analyzer.prompt_logger.save_text_log()
    
    print("\n" + "=" * 60)
    print("Step 3 完了！")
    print(f"出力先: {LLM_ANALYSIS_DIR}")
    print(f"プロンプトログ: {PROMPT_LOG_DIR}")
    print("=" * 60)
    
    return results, analyzer.prompt_logger


if __name__ == "__main__":
    main()

