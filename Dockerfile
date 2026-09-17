FROM python:3.13-slim

LABEL org.opencontainers.image.title="llm-ops-toolbox"
LABEL org.opencontainers.image.description="Maintenance toolbox for the llm-ops GGUF model library: upstream update checks, integrity verification, and Hugging Face downloads"
LABEL org.opencontainers.image.source="https://github.com/hurui200320/llm-ops"

COPY scripts/*.py /opt/llm-toolbox/

RUN chmod 755 /opt/llm-toolbox/*.py \
 && ln -s /opt/llm-toolbox/check_updates.py /usr/local/bin/check_updates \
 && ln -s /opt/llm-toolbox/verify_checksums.py /usr/local/bin/verify_checksums \
 && ln -s /opt/llm-toolbox/download_hf.py /usr/local/bin/download_hf

ENV PYTHONUNBUFFERED=1
ENV LLM_LIBRARY_ROOT=/data
WORKDIR /data

CMD ["bash"]
