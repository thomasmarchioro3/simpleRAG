
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

def get_embeddings():
    # TODO: Add support for Bedrock
    
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")

    embeddings = get_embeddings()
    print(embeddings.embed_query("The fish twisted and turned").__len__())