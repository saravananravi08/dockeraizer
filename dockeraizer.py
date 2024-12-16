import streamlit as st
from pathlib import Path
from typing import Dict, List
from litellm import completion


def read_important_files(directory_path: str) -> Dict[str, str]:
    """
    Read contents of important configuration files in the project directory.

    Args:
        directory_path (str): Path to the project directory

    Returns:
        Dict mapping filenames to their contents
    """
    important_files = [
        # Python
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "Pipfile",
        # Node.js
        "package.json",
        "package-lock.json",
        # Java
        "pom.xml",
        "build.gradle",
        # Go
        "go.mod",
        # Rust
        "Cargo.toml",
        # Docker existing configs
        "Dockerfile",
        "docker-compose.yml",
        # Generic configs
        "config.json",
        "config.yaml",
        "config.yml",
    ]

    found_files = {}
    path = Path(directory_path)

    for filename in important_files:
        file_path = path / filename
        if file_path.is_file():
            try:
                content = file_path.read_text()
                found_files[filename] = content
            except Exception as e:
                found_files[filename] = f"Error reading file: {str(e)}"

    return found_files


def generate_directory_markdown(directory_path: str, indent_level: int = 0) -> str:
    """Generate a markdown representation of a directory structure."""
    path = Path(directory_path)
    markdown = ""

    try:
        items = [item for item in path.iterdir() if not item.name.startswith(".")]
        items = sorted(items, key=lambda x: (not x.is_dir(), x.name.lower()))

        for item in items:
            indent = "    " * indent_level
            if item.is_dir():
                markdown += f"{indent}📁 **{item.name}/**\n"
                markdown += generate_directory_markdown(str(item), indent_level + 1)
            else:
                extension = item.suffix.lower()
                if extension in [".py", ".java", ".cpp", ".js"]:
                    emoji = "📜"
                elif extension in [".jpg", ".png", ".gif", ".bmp"]:
                    emoji = "🖼️"
                elif extension in [".pdf", ".doc", ".docx", ".txt"]:
                    emoji = "📄"
                else:
                    emoji = "📋"
                markdown += f"{indent}{emoji} {item.name}\n"

    except PermissionError:
        markdown += f"{indent}❌ *Access Denied*\n"
    except Exception as e:
        markdown += f"{indent}❌ *Error: {str(e)}*\n"

    return markdown


def extract_docker_files(response: str) -> tuple:
    """Extract Dockerfile, docker-compose.yml content and summary from the response."""
    dockerfile = ""
    docker_compose = ""
    summary = ""

    blocks = response.split("```")

    # Extract summary (assumes it's before any code blocks)
    if blocks and not blocks[0].strip().lower().startswith("dockerfile"):
        summary = blocks[0].strip()
        blocks = blocks[1:]

    blocks = [block for block in blocks if block.strip()]

    for block in blocks:
        if block.strip().lower().startswith("dockerfile"):
            dockerfile = block.replace("dockerfile", "").strip()
        elif block.strip().lower().startswith(("yaml", "docker-compose")):
            docker_compose = (
                block.replace("docker-compose.yml", "").replace("yaml", "").strip()
            )

    return dockerfile, docker_compose, summary


def stream_response(directory_path: str, model: str, api_key: str):
    """Stream responses from the AI model."""
    # Generate directory structure
    directory_markdown = generate_directory_markdown(directory_path)

    # Read important configuration files
    config_files = read_important_files(directory_path)

    # Create the prompt with all available information
    config_files_text = "\n\n".join(
        f"=== {filename} ===\n{content}" for filename, content in config_files.items()
    )

    complete_prompt = f"""
Directory Structure:
{directory_markdown}

Configuration Files Found:
{config_files_text}
"""

    messages = [
        {
            "role": "system",
            "content": """You are an experienced DevOps engineer. Based on the provided directory structure and configuration files:
            1. First provide a brief summary of the project and your containerization approach (2-3 sentences)
            2. Create an appropriate Dockerfile optimized for the project
            3. Create a docker-compose.yml file with necessary services and configurations
            
            Start with the summary in plain text, then provide the Dockerfile and docker-compose.yml files in markdown code blocks with appropriate labels.""",
        },
        {"role": "user", "content": complete_prompt},
    ]

    try:
        response = completion(
            model=model, messages=messages, stream=True, api_key=api_key
        )

        full_response = ""
        for part in response:
            chunk = part.choices[0].delta.content
            if chunk:
                full_response += chunk
                yield chunk

        if full_response:
            dockerfile, docker_compose, summary = extract_docker_files(full_response)
            if "docker_files" not in st.session_state:
                st.session_state.docker_files = []
            st.session_state.docker_files.append(
                {
                    "timestamp": st.session_state.get("message_count", 0),
                    "dockerfile": dockerfile,
                    "docker_compose": docker_compose,
                    "summary": summary,
                }
            )
            st.session_state.message_count = (
                st.session_state.get("message_count", 0) + 1
            )

    except Exception as e:
        yield f"Error during streaming: {str(e)}"


def main():
    st.set_page_config(
        page_title="DockerAizer - Instant Container Configuration Generator",
        page_icon="🐳",
        layout="wide",
    )

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docker_files" not in st.session_state:
        st.session_state.docker_files = []
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0

    # Sidebar for API configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        api_key = st.text_input("API Key", type="password")

        # Model selection section
        st.markdown("#### Model Selection")

        # Selection method
        model_selection_method = st.radio(
            "Choose model selection method:",
            ["Common Models", "Custom Model"],
            help="Select from common models or enter a custom LiteLLM model name",
        )

        if model_selection_method == "Common Models":
            model = st.selectbox(
                "Select Model",
                [
                    "openai/gpt-4o",
                    "openai/gpt-4o-mini",
                    "anthropic/claude-3-opus",
                    "anthropic/claude-3-sonnet",
                    "mistral/mistral-large-latest",
                    "mistral/mistral-medium",
                    "google/gemini-pro",
                ],
            )
        else:
            st.markdown(
                """
            Enter any LiteLLM-supported model name. Examples:
            - `anthropic/claude-3-opus`
            - `azure/gpt-4`
            - `aws/bedrock/claude`
            
            [View all supported models →](https://docs.litellm.ai/docs/providers)
            """
            )
            model = st.text_input(
                "Custom Model Name",
                placeholder="Enter LiteLLM model name",
                help="Enter any model name supported by LiteLLM",
            )

        # Display selected model
        st.markdown("---")
        st.markdown(f"**Selected Model:** `{model}`")

    st.title("🐳 DockerAizer")
    st.markdown(
        "### Transform Any Project into Container-Ready Configuration in Seconds"
    )

    # New detailed introduction with instructions and caution
    st.markdown(
        """
    #### 🚀 Getting Started
    Paste your project's directory path below to start the containerization process.
    
    ⚠️ **Important Note:**
    - The generated configurations are AI-assisted suggestions
    - All outputs should be reviewed and tested before deployment
    - Manual adjustments may be needed based on your specific requirements
    - Always test configurations in a development environment first
    """
    )

    directory_path = st.text_input(
        "Enter the directory path:",
        value="",
        key="directory_input",
    )

    if directory_path:
        if not api_key:
            st.error("Please enter your API key in the sidebar first!")
            return

        path = Path(directory_path)
        if not path.exists():
            st.error("❌ Directory does not exist! Please provide a valid path.")
        else:
            # Show directory structure
            with st.expander("View Directory Structure"):
                st.markdown(
                    f"```markdown\n{generate_directory_markdown(directory_path)}\n```"
                )

            # Show found configuration files
            config_files = read_important_files(directory_path)
            with st.expander("View Found Configuration Files"):
                for filename, content in config_files.items():
                    st.text(f"Found: {filename}")

            # Generate Docker configurations
            st.subheader("🔍 Analysis & Recommendations")
            with st.chat_message("assistant"):
                st.write_stream(stream_response(directory_path, model, api_key))


if __name__ == "__main__":
    main()
