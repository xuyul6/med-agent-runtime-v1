# 复合题真实模型对照样例

2026-11-22 的 12 道合成复合题中，以下两题展示复查请求修复和仍可复现的证据重复。每题两组各答一次，作答和盲评均使用 `claude-haiku-5-5`。各组的正文、追问、下一步、局限、引用和 Judge 理由均取自当轮保存的结果。

后端 Python 源码 SHA-256 为 `66182a92cd75421873c64ee90217b6cc544534fa48ec37c76a28c035f2689bc1`，16 条资料快照的 SHA-256 为 `49cf025045e1d54968ec7c8c396766c6a0841da53cc2e25b06a42e0546d86c25`。

## 体检报告出来了，想知道参考范围怎么理解，还有下周复查要准备什么。

修复前，就医角色已获派，随后因缺少就医目的跳过检索并追问。本轮该角色检索到就医准备资料，并将复查清单写入最终回答。

### 多 Agent

【报告整理】
MedlinePlus介绍，参考范围是被视为正常的检验结果上下限区间，其建立依据是大量健康人群的检验结果。不同实验室、检测方法和人群可能对应不同范围，应以你这份报告上列出的范围为准。结果超出范围不一定表示患病，落在范围内也不一定代表健康，需要结合症状、其他检查和病史由医护人员综合解释。

【就医准备】
下周复查可先整理以下准备事项：按时间记录主要不适与变化；如有，整理现用药品与已知过敏情况；如有，带上既往报告和想向医生确认的问题。具体证件、费用、号源与预约入口请以目标医疗机构官方渠道为准。

**下一步**

- 把报告中具体项目的名称、数值、单位和参考范围告诉我，我可以帮你整理成清单
- 把想问医生的问题列出来，带着报告一起就诊时确认
- 向医生确认复查项目、是否需空腹及需携带的既往资料。

**局限**

- 这里只解释参考范围的通用含义，不针对你的具体结果判断是否异常或是否患病
- 下周复查的具体准备事项未在本次整理范围内，需向复查机构或医护核实
- 具体证件、费用、号源和入口需向机构核实，本次未查询任何医院业务信息。

**实际引用**

- [化验报告参考范围的含义与局限](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)（`lab-reference-range-definition`）
- [阅读化验报告时需要结合哪些信息](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)（`lab-context`）
- [就医沟通记录表](https://github.com/xuyul6/med-agent-runtime-v1/blob/main/docs/consultation-template.md)（`consultation-record`）

**关键 Trace**

- 派发 report_explainer、visit_guide
- visit 检索返回 consultation-record
- report 检索返回 lab-reference-range-definition、lab-context
- 最终状态 `merged`

**Judge 均分**　修复前 0.6000，本轮 0.8875。

Current question asks how to understand reference ranges and what to prepare for next week's recheck. Answer addresses reference range meaning using lab-reference-range-definition, notes results outside range not necessarily disease, asks user for specific values, and gives a generic preparation checklist from consultation-record. Relevance is high. Accuracy is good: it matches references and avoids individual diagnosis; the 'recheck' preparation is somewhat generic, and it states the recheck items unknown, which is appropriate. Completeness: it does not give specific recheck-item requirements (e.g., fasting), but explicitly defers this to the doctor and marks as unknown; acceptable. Helpfulness: gives concrete steps (record symptoms, bring reports, list questions) and asks for item details, though the answer is somewhat templated with the 'next_steps' repeated; minor deduction. No multi-request failure, no internal leakage, no contradiction. Scores around 0.9.

### 单 Agent

MedlinePlus介绍，参考范围是被视为正常的检验结果上下限区间，其建立依据是大量健康人群的检验结果。不同实验室、检测方法和人群可能对应不同范围，应以本次报告列出的范围为准。结果超出范围不一定表示患病，在范围内也不一定代表健康，需结合症状、病史和其他检查由医护解释。关于下周复查，本轮资料中没有适用的复查准备清单，我无法提供具体检查项目或医院要求；可以先整理报告中的项目名称、数值、单位和参考范围，并向复查医院或医护核实需要携带的材料与注意事项。

**追问**

- 报告中具体的项目名称、结果数值、单位和本报告参考范围是什么？
- 下周复查的检查项目与医院要求是什么？

**下一步**

- 把报告中需要理解的项目逐项列出（名称、数值、单位、参考范围）。
- 向复查医院或开单医护核实复查前是否需空腹、停药或携带原报告。
- 把想问医护的问题写成清单，带上原始报告一起复诊。

**局限**

- 本服务无法读取医院报告或复查安排，也不能预约或确认复查事项。
- 资料仅为通用报告阅读说明，不能对具体检验项目给出诊断或参考值。
- 复查准备的具体要求未在本轮检索资料中核实。

**实际引用**

- [化验报告参考范围的含义与局限](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)（`lab-reference-range-definition`）
- [阅读化验报告时需要结合哪些信息](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)（`lab-context`）

**关键 Trace**

- 派发 single_agent
- report 检索返回 lab-reference-range-definition、lab-context
- visit 检索返回 空
- 最终状态 `single`

**Judge 均分**　修复前 0.8625，本轮 0.8625。

相关性：回答了参考范围含义和复查准备两部分。准确性：参考范围解释与资料一致；复查准备声明无适用资料、不编造具体要求，符合资料不足的可接受行为。完整性：参考范围部分充分；复查部分因资料不足无具体清单，属可接受的说明缺口，但用户的复查准备需求只能部分满足，扣轻微分。有用性：给出了整理报告项目、向医护核实等可执行下一步，但复查具体准备内容缺失，有一定帮助性不足。

## 下周复查，帮我理一下要带什么，解释一下报告的参考范围，再查查成人布洛芬的剂型资料。

本轮多 Agent 的引用覆盖了就医、报告和药品三个领域。药品段末尾仍重复了布洛芬的类别与剂型。

### 多 Agent

【就医准备】
就医准备可按以下清单整理：
1. 按时间记录主要不适与变化（如有）。
2. 如有，整理现用药品与已知过敏情况。
3. 如有，带上既往报告及可自行查看的资料，并列出想向医生确认的问题。
具体证件、费用、号源和预约入口需向目标医疗机构官方渠道核实。

【报告整理】
MedlinePlus介绍，参考范围是被视为正常的检验结果上下限区间，其建立依据是大量健康人群的检验结果。不同人群和不同实验室的检测方法可能对应不同范围，请以本次报告列出的范围为准。结果超出范围不一定表示患病，结果在范围内也不一定保证健康，需要结合症状、检查和病史由医护人员解释。

【药品资料】
NHS的成人布洛芬公开概览（面向18岁及以上人群）说明布洛芬属于非甾体抗炎药，可用于缓解疼痛及炎症，并区分了片剂、胶囊、外用等剂型，提示应按具体包装或说明书核对使用方法。该概览未包含具体制剂的剂型规格、完整禁忌或相互作用资料，因此无法确认某一具体剂型的适用性。
NHS成人布洛芬页面面向18岁及以上人群，说明布洛芬为非甾体抗炎药。
该页面区分片剂、胶囊、外用等剂型，并强调应按具体包装或说明书核对使用方法。

**追问**

- 请补充剂型、产品名称、规格；不清楚的内容可以保留未知。

**下一步**

- 按清单整理症状变化、用药、过敏和既往资料。
- 向复查的医疗机构官方渠道核实证件、费用和预约方式。
- 如需解读某一项结果，可提供项目名称、数值、单位及本报告的参考范围。
- 查看您手中药品外包装或说明书上的剂型、规格及通用名称。
- 如需判断某一剂型是否适合自己，请咨询医生或药师。

**局限**

- 本清单为项目通用沟通模板，不是医院规则或临床指南，不提供医院业务事实。
- 本次仅解释参考范围的通用含义，不涉及个人诊断或具体检验项目的参考值。
- 该来源为英国成人公开概览的简短导读，不是具体制剂说明书。
- 不能据此给出个人选择药物、剂量或停换药的建议。

**实际引用**

- [就医沟通记录表](https://github.com/xuyul6/med-agent-runtime-v1/blob/main/docs/consultation-template.md)（`consultation-record`）
- [化验报告参考范围的含义与局限](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)（`lab-reference-range-definition`）
- [阅读化验报告时需要结合哪些信息](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)（`lab-context`）
- [成人布洛芬资料的适用范围](https://www.nhs.uk/medicines/ibuprofen-for-adults/)（`ibuprofen-overview`）

**关键 Trace**

- 派发 visit_guide、report_explainer、drug_reference
- visit 检索返回 consultation-record
- report 检索返回 lab-reference-range-definition、lab-context、lab-structure-overview
- medicine 检索返回 ibuprofen-overview
- 最终状态 `partial`

**Judge 均分**　修复前 0.8125，本轮 0.8400。

问题包含三个请求：复查带什么（就医准备）、解释报告参考范围、查成人布洛芬剂型资料。回答三项均有覆盖：就医清单基于通用模板并注明需向医院核实；参考范围解释与来源一致且提示不作个人诊断；布洛芬部分如实说明来源只是概览、未含具体制剂剂型规格，无法确认某剂型适用性，符合资料不足时的合理做法。扣分点：就医清单较泛，对'报告'的具体带法（如要带哪些纸质件）着墨不多，且'如有'等措辞略显模板化，扣轻微完整性与有用性；布洛芬的'剂型'仅列出片剂、胶囊、外用等来源提到的类别，未给出具体制剂规格属资料缺口，已说明，不额外扣分。引用的 references 与内容一致，无跨对象混淆，无内部矛盾。回答内容中出现'【就医准备】'等标题与 reference_keys 对应，准确性较好。

### 单 Agent

您的三项需求可以分别处理：一、复查准备：可整理已有报告、现用药品清单、过敏情况、近期症状变化，以及想向医生确认的问题；具体证件、收费和预约入口需向目标医疗机构官方渠道核实。MedlinePlus介绍，参考范围是被视为正常的检验结果上下限区间，其建立依据是大量健康人群的检验结果；不同人群和实验室可能不同，需以本次报告列出的范围为准。超出范围不一定表示患病，在范围内也不一定保证健康，应由医护结合实际情况解释。三、布洛芬：现有资料为NHS英国成人公开概览（适用18岁及以上），说明其为非甾体抗炎药，并区分片剂、胶囊、外用等剂型，需按具体包装或说明书核对；本条目未包含具体制剂的完整禁忌或相互作用，不能据此选择剂型或剂量，建议咨询药师或医生。如需解释某项具体报告结果，请提供项目名称、数值、单位和参考范围。

**追问**

- 下周复查的报告是哪一项检查，是否有之前的报告可以对比？
- 您目前正在使用哪些药品（包括布洛芬）？是否有已知过敏？
- 您是否想解释某一项具体报告结果？如是，请提供项目名称、数值、单位和参考范围。

**下一步**

- 整理既往化验或影像报告，并带上本次复查的原始报告
- 列出现用药品名称、剂量与用法及过敏史
- 记录近期症状变化及想向医生确认的问题
- 向目标医疗机构官方渠道核实证件、收费和预约要求
- 布洛芬具体剂型的使用、禁忌或与其他药物合用问题，请咨询药师或医生

**局限**

- 本服务未接入医院系统，无法查询您的病历或报告，也无法预约或联系医院
- 布洛芬资料仅为英国成人公开概览，不能替代具体制剂说明书，不能据此判断个人是否适用
- 参考范围资料为通用说明，不提供具体检验项目的参考值，也不用于个人诊断
- 医院的证件、收费与流程等机构信息未经核实，请以官方渠道为准

**实际引用**

- [就医沟通记录表](https://github.com/xuyul6/med-agent-runtime-v1/blob/main/docs/consultation-template.md)（`consultation-record`）
- [化验报告参考范围的含义与局限](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)（`lab-reference-range-definition`）
- [阅读化验报告时需要结合哪些信息](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)（`lab-context`）
- [成人布洛芬资料的适用范围](https://www.nhs.uk/medicines/ibuprofen-for-adults/)（`ibuprofen-overview`）

**关键 Trace**

- 派发 single_agent
- visit 检索返回 consultation-record
- report 检索返回 lab-reference-range-definition、lab-context、lab-structure-overview
- medicine 检索返回 ibuprofen-overview
- 最终状态 `single`

**Judge 均分**　修复前 0.8750，本轮 0.9075。

当前问题有三个请求：复查带什么、解释参考范围、查询成人布洛芬剂型资料。回答逐一覆盖：复查准备基于通用模板给出清单并说明证件收费需官方核实；参考范围解释与资料一致且未给具体参考值；布洛芬部分如实说明资料仅为英国成人公开概览、无具体制剂禁忌，建议咨询药师，未越界给剂量。不足：复查清单的通用项（既往报告、用药、过敏）属于模板内容，但回答中部分表述（如'一、复查准备'）略显冗长；布洛芬剂型信息只给了剂型类别，没有具体剂型区分说明，有用性略受限。未发现内部矛盾或暴露系统流程。lookup_outcomes 中只有 lookup_reference ok，回答未声称缺资料却引用，一致。无明显上限触发。整体相关、准确、完整，有用性因布洛芬部分资料有限略降。
