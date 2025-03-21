# RAG Project with LangChain and Pinecone

Este proyecto implementa un sistema de Retrieval Augmented Generation (RAG) utilizando LangChain y Pinecone en Google Colab. El objetivo es crear una aplicación que pueda responder preguntas basadas en el contenido de un sitio web, utilizando un modelo de lenguaje (LLM) para generar respuestas y un almacén de vectores para recuperar información relevante.

## Requisitos Previos

Antes de ejecutar el proyecto, asegúrate de tener las siguientes dependencias instaladas:

- **Google Colab**: El proyecto está diseñado para ejecutarse en Google Colab.
- **API Keys**: Necesitarás claves API para OpenAI y Pinecone.

## Instalación de Dependencias

El proyecto utiliza varias bibliotecas de Python que deben instalarse antes de ejecutar el código. A continuación se detallan los comandos para instalar las dependencias necesarias:

```bash
!pip install langchain openai langchain-openai langchain-community langchain-text-splitters langchainhub pinecone-client
!pip install -qU langchain-openai
!pip install -qU langchain-pinecone pinecone-notebooks
!pip install langgraph
!pip install bs4
```

## Configuración del Entorno

### Configuración de OpenAI

Para utilizar el modelo de lenguaje de OpenAI, necesitas configurar tu clave API:

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
```

### Configuración de Pinecone

Pinecone se utiliza como almacén de vectores para almacenar y recuperar documentos. Configura tu clave API de Pinecone:

```python
import getpass
import os

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
```

## Componentes del Proyecto

### Modelo de Lenguaje (LLM)

Se utiliza el modelo `gpt-4` de OpenAI como el modelo de lenguaje principal:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4", model_provider="openai")
```

### Modelo de Embeddings

Se utiliza el modelo `text-embedding-3-large` de OpenAI para generar embeddings de los documentos:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Almacén de Vectores (Pinecone)

Pinecone se utiliza como almacén de vectores para almacenar y recuperar documentos:

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-test-index"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
```

### Carga y División de Documentos

Se carga el contenido de un blog y se divide en fragmentos más pequeños para su indexación:

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))},
)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
```

### Indexación de Documentos

Los documentos divididos se indexan en Pinecone:

```python
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
_ = vector_store.add_documents(documents=all_splits)
```

### Construcción del Grafo de Aplicación

Se define un grafo de aplicación utilizando LangGraph para manejar la lógica de recuperación y generación de respuestas:

```python
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
```

### Ejecución de la Aplicación

Finalmente, se ejecuta la aplicación para responder preguntas basadas en el contenido del blog:

```python
response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
```

## Uso del Proyecto

1. **Configuración**: Asegúrate de tener las claves API de OpenAI y Pinecone configuradas.
2. **Ejecución**: Ejecuta las celdas en orden para cargar, indexar y consultar los documentos.
3. **Consulta**: Puedes hacer preguntas sobre el contenido del blog utilizando la función `graph.invoke`.

## Ejemplos de Consultas

- **Pregunta**: "What is Task Decomposition?"
- **Respuesta**: La aplicación generará una respuesta basada en el contenido del blog.

![image](https://github.com/user-attachments/assets/b2f78784-46f3-43a5-a399-b6d594c636ee)


## Visualización del Grafo

Puedes visualizar el grafo de la aplicación utilizando la siguiente función:

![image](https://github.com/user-attachments/assets/03c3f940-d82d-448f-a685-2a5ad092e8f7)

### **Autor**

- Ana Maria Duran - *AREP* *LangChain-RAG-AREP* - [AnaDuranB](https://github.com/AnaDuranB)

---
¡Gracias por revisar este proyecto! 😊
