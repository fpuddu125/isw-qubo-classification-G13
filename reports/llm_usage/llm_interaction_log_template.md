# LLM Interaction Log

This file records all relevant interactions with Large Language Models (LLMs) used during the project.

Each interaction must be recorded as a separate entry. You must use the following unique interaction ID: `G03-06-12-001`, `G03-06-12-002`, etc., where GXX is the group identifier, MM-DD means month and day, followed by a progressive number starting from 001 for the first recording of the day.

Do **not** overwrite previous entries. Do **not** use the same downloaded file name twice. All downloaded files must be stored in the project repository and referenced exactly with their relative path.

The same day must not be covered by more than one interaction log. If you feel that the interaction log is becoming too long, at the end of the day close the log and start another one the following day. The logs are identified by their log_id, with the group code and a progressive number NN: "GXX-NN". Each log must be contained in a file named: "LOG-GXX-NN.md". 

---

## Metadata

```yaml
group_id: "GXX"
repository_url: "https://github.com/..."
students:
  - matricola: "60/61/XXXXX"
    name: "Student 1"
  - matricola: "60/61/XXXXX"
    name: "Student 2"
last_update: "YYYY-MM-DD"
log_id: "GXX-NN"
```

---

# Interaction Entries

---

## Interaction GXX-MM-DD-001

### 1. LLM and chat information

```yaml
llm_name: "ChatGPT / Claude / Gemini / Copilot / Cursor / DeepSeek / Other"
llm_version_or_model: "e.g. GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro"
chat_name_or_identifier: "Short name or identifier of the active chat"
interaction_mode: "web_chat / IDE_assistant / API / desktop_app / other"
```

### 2. Author of the interaction

```yaml
performed_by: "couple"   # use "couple" or the matricola of the individual student
```

### 3. Project phase

Select one or more phases.

```yaml
project_phase:
  - "requirements_understanding"
  - "preprocessing"
  - "feature_selection_qubo"
  - "optimization_algorithm"
  - "classification_model"
  - "testing"
  - "debugging"
  - "documentation"
  - "gui"
  - "repository_structure"
  - "other"
```

### 4. Input files and/or context provided to the LLM

List any files or project documents provided to the LLM.
Record ("yes"/"no") if you provided code snippets, error messages, dataset excerpts.
If the prompt was itself generated or improved by another LLM interaction, mention the source interaction ID.

```yaml
input_context:
  files_uploaded:
    - file_name: null
      repository_path: null
      description: null
  code_snippets: null
  error_messages: null
  dataset_excerpts: null
  prompt_generated_by: null 
```

### 5. Student prompt

Paste the exact prompt sent to the LLM. 

```text
Paste the full prompt here.
```

### 6. LLM response

Paste the full response received from the LLM. If the response is very long, it is acceptable to store it in a separate file and reference it here.

```text
Paste the full LLM response here.
```

If the full response is stored in a separate file:

```yaml
response_stored_in_file:
  used: false
  file_name: null
  repository_path: null
```

### 7. Files generated or downloaded from the LLM response

List every file produced by the LLM and downloaded or copied into the project. Each file name must be unique across the whole repository.

```yaml
generated_or_downloaded_files:
  - unique_file_name: null
    repository_path: null
    file_type: null       # e.g. py, ipynb, md, yaml, csv, json, txt
    created_from_response: true
    short_description: null
    referenced_in_response: true
```

### 8. How the LLM output was used

```yaml
usage_of_output:
  used_without_changes: null  # true / false
  modified_before_use: null   # true / false
  description_of_modifications: null
  related_repository_files:
    - null
```

### 9. Problems, errors, or hallucinations

```yaml
issues_found:
  any_issue_found: null    # "yes"/"no"  The subsequent part only if "yes"
  issue_categories:    # Delete non-pertinent ones
    - "none"
    - "syntax_error"
    - "runtime_error"
    - "wrong_algorithm"
    - "wrong_qubo_formulation"
    - "wrong_library_usage"
    - "hallucinated_function_or_api"
    - "poor_explanation"
    - "security_or_privacy_issue"
    - "other"
  description: null
  how_issue_was_resolved: null
```

### 10. Usefulness and reliability assessment

Use a scale from 1 to 5 (1 = lowest grade, 5 = highest grade).

```yaml
assessment:
  usefulness_1_to_5: null
  correctness_1_to_5: null
  clarity_1_to_5: null
  confidence_after_verification_1_to_5: null
  would_reuse_this_output: null
  notes: null
```

---

## Interaction  GXX-MM-DD-002

Copy the structure of `GXX-MM-DD-001` for every new interaction.

---

# Suggested file naming convention

Use unique and descriptive names for every file generated or downloaded from an LLM response.

Recommended format:

```text
LLM_<interaction_id>_<short_description>.<extension>
```

Examples:

```text
LLM_G03-06-05-002_preprocessing_function.py
LLM_G03-06-12-001_qubo_cost_function.py
LLM_G03-06-12-001_test_feature_selection.py
LLM_G03-06-13-003_readme_section.md
```

---

# Final checklist

Before submission, verify that:

- [ ] Every relevant LLM interaction has been recorded.
- [ ] Each interaction has a unique ID.
- [ ] Date and time are present for every interaction.
- [ ] The author is identified as either `couple` or one student matricola.
- [ ] The full prompt is included.
- [ ] The full LLM response is included or correctly referenced as a separate file.
- [ ] Every downloaded/generated file has a unique name, is present in the repository and is correctly referenced in this log.
- [ ] Problems, errors, and hallucinations have been reported honestly.
