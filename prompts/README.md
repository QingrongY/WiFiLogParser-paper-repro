# Complete experimental prompts

This directory indexes the complete prompt templates used by every LLM-based
component reported in the paper. Runtime log samples, validation diagnostics,
and parser examples replace the brace-delimited placeholders shown in these
templates.

| Experimental component | Prompt text | Implementation source |
| --- | --- | --- |
| Timestamp-header induction | [`timestamp_header.txt`](timestamp_header.txt) | [`timestamp_agent.py`](../src/apps_v2/logparser/services/log_extractor/timestamp_agent.py) |
| Timestamp-header refinement | [`timestamp_header_refinement.txt`](timestamp_header_refinement.txt) | [`timestamp_agent.py`](../src/apps_v2/logparser/services/log_extractor/timestamp_agent.py) |
| WiFiLogParser content parsing | [`wifilogparser_content_parsing.txt`](wifilogparser_content_parsing.txt) | [`llm_parser.py`](../src/apps_v2/logparser/services/log_extractor/llm/llm_parser.py) |
| WiFiLogParser self-repair | [`self_repair.txt`](self_repair.txt) | [`llm_repairer.py`](../src/apps_v2/logparser/services/log_extractor/fallback/llm_repairer.py) |
| Direct LLM parsing | [`direct_llm.txt`](direct_llm.txt) | Appendix A of the paper |
| LILAC+/LogBatcher+ event labeling | [`classify_template.txt`](../baselines/prompts/classify_template.txt) | First post-processing stage for both template baselines |
| LILAC+/LogBatcher+ AP/client field extraction | [`extract_fields.txt`](../baselines/prompts/extract_fields.txt) | Second post-processing stage for event templates |

The two files under `baselines/prompts/` were later organized and versioned in
the reproducibility repository, but their text is the same as the prompts used
to produce the LILAC+ and LogBatcher+ results reported in Table 5. The separate
2,000-line reproducibility rerun also uses these prompts, but its results are
not used in the manuscript.
