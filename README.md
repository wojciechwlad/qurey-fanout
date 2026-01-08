# Gemini Query Fanout (Streamlit)

A Streamlit GUI that replicates the logic from `Query_fanout.ipynb`:

- Generates 10 semantically related questions using Gemini generate API
- Uses Gemini embeddings to compute cosine similarities and rank questions
- Allows downloading the top results as a JSON file

## Features

### 🤖 Model Selection
- Choose from available Gemini models (default: `gemini-3-flash-preview`)
- Refresh button (🔄) to fetch latest models from Gemini API
- Supports models: gemini-3-flash-preview, gemini-2.0-flash-exp, gemini-1.5-flash, gemini-1.5-pro, and more

### 🔑 API Key Management
- **Save** (💾) - Save your API key locally to `.gemini_api_key` file
- **Load** (📂) - Load previously saved API key (auto-loads on startup)
- **Delete** (🗑️) - Remove saved API key file
- API key stored with base64 encoding (not encryption - keep project folder secure)

### ⚙️ Generation Configuration
Customize the AI generation parameters:
- **Temperature** (0-1): Controls randomness - lower is more focused, higher is more creative (default: 0.7)
- **Top K** (1-100): Limits sampling to top K tokens (default: 40)
- **Top P** (0-1): Nucleus sampling probability threshold (default: 0.95)
- **Max Output Tokens** (1-8196): Maximum length of generated response (default: 2048)

### 🌍 Language Support
- Specify any language for keyword analysis and output
- Default: Polish

## Quick start (using `uv`)

1. Install `uv` (see https://docs.astral.sh/uv/)
2. In the project folder:

```bash
uv add streamlit requests numpy
uv run streamlit run app.py
```

## Quick start (pip)

```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\\Scripts\\activate` on Windows
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. **Enter a keyword** to analyze
2. **Select language** (default: Polish)
3. **Enter or load API key** (save for future sessions)
4. **Choose AI model** (refresh to get latest models)
5. **Adjust Top N results** slider (1-10 questions)
6. **Optional**: Expand Generation Config to customize AI parameters
7. **Click Generate** to create query fanout
8. **Download results** as JSON

## Security Notes

- API key can be saved locally in `.gemini_api_key` (base64 encoded)
- This file is included in `.gitignore` to prevent accidental commits
- **Important**: Keep your project folder secure as the key is stored locally
- The API key is sent to Google's Gemini API endpoints for generation and embedding requests

## Notes

- If you want to use a project environment manager, `uv` is supported and recommended for faster installs.
- Model list is cached for 5 minutes to reduce API calls
- Embeddings and query results are cached to improve performance
