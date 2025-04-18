FROM python:3.9

WORKDIR /app


COPY requirements.txt .
RUN pip install scikit-learn numpy pandas matplotlib seaborn tensorflow networkx gym==0.26.0
RUN pip install -r requirements.txt

RUN pip uninstall -y torch torchvision torchaudio
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

ENV PATH_SIMULATION=/app
ENV PYTHONUNBUFFERED=1

# Copiar arquivos do projeto
COPY . .

RUN sed -i 's|c:/Users/Utilizador/Ambiente de Trabalho/Tese/RRC_DRL_Update/RRC_DRL_Updates|/app|g' environmental_variables.py

# Comando para executar o script de combinações
CMD ["python", "./run_all_combinations.py"]