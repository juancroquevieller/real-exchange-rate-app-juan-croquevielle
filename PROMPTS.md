# AI assistance record

Date: October 25, 2026 (America/Chicago); work may continue October 26.

Assistant: OpenAI Codex, GPT-6-based agent. Exact model variant is not exposed in the conversation context; confirm the name displayed in the app before final submission.

## User request (faithful summary)

Build an app in my GitHub environment where users select two countries and compare nominal and real exchange rates. Support daily, monthly and annual views with appropriate date controls. Put the assignment's graph first, with Real/Nominal/Both, separate axes and two simple, insightful bullets. Add a second graph with a stable RER benchmark grounded in a defensible period, the Eichenbaum/Johannsen/Rebelo paper and the prior RER homework. Omit unsupported benchmarks and explain why. Generate valuation insights and conditional actions tailored to traders and companies with foreign loans or investments. Use a compact, attractive white interface responsive to phones, tablets, laptops and big monitors.

## Course constraints incorporated

Streamlit / Python 3.12; monthly IMF CPI and FX from 1976 where available; S as Country 2 currency per Country 1 unit; RER = S × CPI1/CPI2; real index = 100 × RER / selected-sample mean; nominal in actual units; Both on separate axes; CSV of inputs and calculations; UK/US direction and average-100 verification; publish code on GitHub and app on Streamlit.

## Implementation decisions for human review

Monthly IMF is the default. Daily is an explicit ECB FX / lagged IMF CPI proxy. The first graph's normalization is separate from the second's historical valuation anchor. Automatic anchors use disclosed conservative screens and published historical regime coverage. EUR/USD January 1999 is an optional course assumption. Deterministic insights make reasoning inspectable and require no AI API key.
