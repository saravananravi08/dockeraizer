# 🐳 DockerAizer

Transform your projects into container-ready configurations using AI-powered analysis.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install streamlit litellm pathlib
```

## Running the App

1. Start the application:
```bash
streamlit run dockeraizer.py
```

2. Open your browser to `http://localhost:8501`

## How to Use

1. **API Configuration**
   - Enter your API key in the sidebar for your chosen LLM provider

2. **Model Selection**
   You have two options for selecting a model:

   a) **Common Models Dropdown**
   - Choose "Common Models" radio button
   - Select from popular models like:
     - OpenAI (gpt-4, gpt-3.5-turbo)
     - Anthropic (claude-3-opus, claude-3-sonnet)
     - Mistral (mistral-large, mistral-medium)
     - Google (gemini-pro)

   b) **Custom Model Input**
   - Choose "Custom Model" radio button
   - Enter any LiteLLM-supported model identifier
   - Examples:
     ```
     azure/gpt-4
     aws/bedrock/claude
     anthropic/claude-3-opus
     ```

3. **Generate Docker Configs**
   - Paste your project's directory path
   - Review the analysis and generated configurations

## Supported Project Types

The tool can analyze projects containing:
- Python files (requirements.txt, setup.py)
- Node.js files (package.json)
- Java files (pom.xml, build.gradle)
- Go files (go.mod)
- Rust files (Cargo.toml)
- Generic configs (yaml, json)

## ⚠️ Important Notes

- Generated configurations are AI suggestions and need human review
- Always test configurations before deployment
- Keep your API keys secure
- Verify model compatibility with your API key
- Configurations may need adjustments for your specific needs

## LiteLLM Model Support

For a complete list of supported models and providers, visit:
[LiteLLM's Provider Documentation](https://docs.litellm.ai/docs/providers)